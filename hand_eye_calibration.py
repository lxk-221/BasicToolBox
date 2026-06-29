#!/usr/bin/env python3
"""
eye-in-hand 手眼标定 (相机装在末端, 标定板固定) —— 棋盘格。

命名遵循 OpenCV: T_A2B 表示把 A 系的点变到 B 系, 即 p_B = T_A2B @ p_A。
    FK        -> T_gripper2base  (每个位姿不同)
    solvePnP  -> T_target2cam    (每个位姿不同)
    待求 X    =  T_cam2gripper   (相机在末端的固定位姿)
恒等式(标定板固定于 base):
    p_base = T_gripper2base @ T_cam2gripper @ T_target2cam @ p_target
    =>  T_gripper2base @ X @ T_target2cam = T_target2base = 常量   (残差据此验证)

运行:
    source /opt/ros/jazzy/setup.bash && source ~/lbot_ws/install/setup.bash
    source ~/xyz_bak/grasp_nuts_env/bin/activate
    cd ~/xyz_bak && python hand_eye_calibration.py
操作: 移动机械臂到不同姿态 -> [c] 采集 (>=10 组, 多变 roll/pitch/yaw)
      -> [s] 求解保存 -> [q] 退出
"""
import os
import time
from collections import deque
from turtle import position
for _p in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_p, None)            # 防代理干扰 Orbbec USB

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lx_useful import LBotArm
from lx_useful.lbot_client import shutdown
from utils import frame_to_bgr_image
from pyorbbecsdk import Config, Context, OBSensorType, Pipeline

# ---------------- 配置 ----------------
PATTERN = (7, 10)          # 棋盘内角点 (列,行); calib.io 8x11 板 = 7x10
SQUARE_MM = 15.0           # 方格边长 mm (决定平移真实尺度)
EULER_SEQ = "xyz"          # FK 欧拉角约定 (lbot_arm.py:36 验证: 内蕴 xyz)
MIN_SAMPLES = 10
OUT_DIR = "./output"
WIN_NAME = "hand_eye (c=采集 s=求解 r=单步复现 q=退出)"   # 实时预览窗口名 (按键需在此窗口聚焦时生效)
CRASH_DUMP = os.path.join(OUT_DIR, "crash_recovery_joints.npz")  # 崩溃时持久化的 joints
HAND_EYE_NPZ = os.path.join(OUT_DIR, "hand_eye.npz")              # [r] 单步复现的数据源


# ---------------- 工具 ----------------
def T_from_fk(pos, euler):
    """FK 的 pos(m) + euler(rad) -> 4x4 T_gripper2base (平移单位 m).
    注意: euler 分解约定依赖 EULER_SEQ, 在 pitch≈±90° (万向锁) 时不可靠。"""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler(EULER_SEQ, list(euler), degrees=False).as_matrix()
    T[:3, 3] = pos
    return T


def T_from_pose_quat(pos, quat_xyzw):
    """位姿的 pos(m) + 四元数(xyzw) -> 4x4 T_gripper2base.
    四元数约定无关, 是 ground truth, 优先使用 (绕开 euler 万向锁/约定歧义)。"""
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(list(quat_xyzw)).as_matrix()
    T[:3, 3] = pos
    return T


def _se3(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def make_obj_points(pattern=PATTERN):
    """标定板角点在 target 系下的 3D 坐标 (mm, Z=0). pattern=(列,行) 决定角点排列顺序。"""
    nx, ny = pattern
    p = np.zeros((nx * ny, 3), np.float32)
    p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    return p * SQUARE_MM


def detect_chessboard(gray):
    """检测棋盘角点 (自动试横向/纵向). 成功返回 (corners, pattern_used), 否则 (None, None).
    返回实际检测用的 pattern, 因为 objectPoints 的角点排列顺序依赖它 (否则 2D/3D 错位)。"""
    for ps in (PATTERN, (PATTERN[1], PATTERN[0])):
        ok, c = cv2.findChessboardCornersSB(gray, ps, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok and len(c) == ps[0] * ps[1]:
            c = cv2.cornerSubPix(
                gray, c, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
            return c, ps
    return None, None


# ---------------- 求解 (核心) ----------------
def solve_hand_eye(samples, K, dist):
    """
    samples: list of dict, 每个含:
        'T_gripper2base' (4x4, 来自 FK, 平移 m)
        'corners'        (像素角点, N×1×2)
    返回: X = T_cam2gripper (4x4, 平移 mm), 残差统计 (n, rot_mean/max_deg, trans_mean/max_mm)
    """
    R_gripper2base, t_gripper2base = [], []   # T_gripper2base, 拆成 R/t 喂给 calibrateHandEye
    R_target2cam,   t_target2cam   = [], []   # T_target2cam
    Tg2b_mm_list = []                         # 单位统一为 mm 后的 T_gripper2base (残差用)
    for s in samples:
        # 每个 sample 用它自己检测时的 pattern 生成 objectPoints (方向必须一致, 否则错位)
        pattern = s.get("pattern", PATTERN)   # 兼容旧数据 (无 pattern 字段时用默认)
        objp = make_obj_points(pattern)
        ok, rv, tv = cv2.solvePnP(objp, s["corners"], K, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        Rt2c, _ = cv2.Rodrigues(rv)                       # T_target2cam 的旋转
        Tg2b = s["T_gripper2base"].copy()
        Tg2b[:3, 3] *= 1000.0                             # FK 平移 m -> mm, 与标定板统一
        R_gripper2base.append(Tg2b[:3, :3])
        t_gripper2base.append(Tg2b[:3, 3])
        R_target2cam.append(Rt2c)
        t_target2cam.append(tv.reshape(3))
        Tg2b_mm_list.append((Tg2b, Rt2c, tv.reshape(3)))

    # eye-in-hand AX=XB: 输出 = T_cam2gripper (即我们要求的 X)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)
    X = _se3(R_cam2gripper, t_cam2gripper.reshape(3))     # T_cam2gripper

    # 残差: 各帧 T_target2base = T_gripper2base @ X @ T_target2cam 应一致
    Ttb = np.stack([Tg2b @ X @ _se3(Rc, tc) for (Tg2b, Rc, tc) in Tg2b_mm_list])
    Rm = Rot.from_matrix(Ttb[:, :3, :3]).mean().as_matrix()
    tm = Ttb[:, :3, 3].mean(0)
    rot = np.array([np.degrees(np.arccos(np.clip(
        (np.trace(Rm.T @ t[:3, :3]) - 1) / 2, -1, 1))) for t in Ttb])
    trans = np.linalg.norm(Ttb[:, :3, 3] - tm, axis=1)
    return X, (len(Ttb), rot.mean(), rot.max(), trans.mean(), trans.max())


# ---------------- 相机 (常驻 pipeline + 出厂内参) ----------------
def init_camera():
    if Context().query_devices().get_count() == 0:
        raise RuntimeError("未发现 Orbbec 相机")
    pl, cfg = Pipeline(), Config()
    cfg.enable_stream(pl.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                      .get_default_video_stream_profile())
    cfg.enable_stream(pl.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                      .get_default_video_stream_profile())
    pl.enable_frame_sync(); pl.start(cfg)
    K = dist = None
    for _ in range(20):                                   # 取一帧让 SDK 解析内参
        fs = pl.wait_for_frames(1000)
        if fs:
            i = pl.get_camera_param().rgb_intrinsic
            d = pl.get_camera_param().rgb_distortion
            K = np.array([[i.fx, 0, i.cx], [0, i.fy, i.cy], [0, 0, 1]], np.float64)
            #dist = np.array([d.k1, d.k2, d.p1, d.p2, d.k3], np.float64)
            dist = np.array([0, 0, 0, 0, 0], np.float64)
            break
    if K is None:
        raise RuntimeError("取不到出厂内参")
    
    cali_K = [
        [695.262784, 0, 637.784594],
        [0, 696.316815, 366.983073],
        [0, 0, 1],
        ]
    cali_dist = [0.00737546861, -0.0291468683, 0.00396127011, 0.0000473029275, 0.0147010447]
    K = np.array(cali_K)
    dist = np.array(dist)
    return pl, K, dist


def grab_color(pl):
    for _ in range(30):
        fs = pl.wait_for_frames(1000)
        if fs and fs.get_color_frame() is not None:
            im = frame_to_bgr_image(fs.get_color_frame())
            if im is not None:
                return im
    return None


# ---------------- 崩溃恢复: 持久化 joints ----------------
def dump_samples_on_crash(samples):
    """崩溃/中断时把已采集样本的 joints 存盘, 下次可一键复现位姿。"""
    if not samples:
        return
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        joints = np.array([s["joints"] for s in samples], dtype=np.float64)
        np.savez(CRASH_DUMP, joints=joints, n_samples=len(joints))
        print(f"[崩溃恢复] 已保存 {len(samples)} 组 joints -> {CRASH_DUMP}")
        print(f"           下次启动按 [r] 可用机械臂自动复现这些位姿")
    except Exception as e:
        print(f"[崩溃恢复] 保存失败: {e}")


def collect_at_current_pose(arm, pl, samples, idx):
    """在当前位姿采集一个样本: 取当前位姿(优先四元数) -> 取帧 -> 检测棋盘 -> 推入 samples。
    成功返回新的 idx, 失败也返回 idx (已自增失败计数)。"""
    # 关键: 主动 spin 累积 ~0.3s 刷新订阅缓存。否则 get_pose/get_joints 只在首次拿到数据,
    # 之后返回陈旧位姿 (lbot_client 的回调仅在 spin 时触发, 而 get_pose 内部仅当 _pose_msg
    # 为 None 才 spin)。主循环只 cv2.waitKey 不 spin, 回调一直不触发, 位姿会锁死在初始值。
    for _ in range(30):
        arm.spin_once(0.01)
    joints = arm.get_joints().positions
    print("current joints: ", joints)
    # 优先用四元数位姿 (约定无关, 绕开 euler 万向锁); 失败则回退 FK
    try:
        pose = arm.get_pose()
        print("current position: ", pose.position)
        print("current orientation: ", pose.orientation)
        T_g2b = T_from_pose_quat(pose.position, pose.orientation)
        rot_desc = "quat"
    except Exception as e:
        pos, euler = arm.forward_kinematics(list(joints))
        T_g2b = T_from_fk(pos, euler)
        rot_desc = f"fk(euler,{EULER_SEQ})"
        print(f"  [提示] get_pose 失败({e}), 回退 FK (euler 约定可能不准)")
    bgr = grab_color(pl)
    if bgr is None:
        print("  取帧失败"); return idx + 1
    corners, pattern_used = detect_chessboard(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if corners is None:
        cv2.imwrite(f"{OUT_DIR}/fail_{idx}.png", bgr)
        print(f"  未检测到棋盘 -> {OUT_DIR}/fail_{idx}.png"); return idx + 1
    samples.append({
        "T_gripper2base": T_g2b,
        "corners": corners,
        "pattern": pattern_used,        # 记录检测用的方向, solve 时生成对应的 objectPoints
        "joints": np.array(joints),
    })
    vis = bgr.copy(); cv2.drawChessboardCorners(vis, pattern_used, corners, True)
    cv2.imwrite(f"{OUT_DIR}/sample_{idx:03d}.png", vis)
    # 用四元数算 RPY 仅作显示 (约定无关的稳定显示)
    rpy_show = np.degrees(Rot.from_matrix(T_g2b[:3,:3]).as_euler("xyz"))
    print(f"  OK #{len(samples)} [{rot_desc}] rpy_deg(xyz)={rpy_show.round(0)} "
          f"board={pattern_used}")
    return idx + 1


# ---------------- 主循环 ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== eye-in-hand 手眼标定 ===")
    arm = LBotArm(namespace="robot1", arm="right", speed=0.2, accel=0.2)
    arm.set_emergency_stop(False); arm.set_enable(True)
    pl, K, dist = init_camera()
    print("K =\n", np.array2string(K, precision=2))

    samples, idx = [], 0
    exited_normally = False
    # [r] 单步复现队列: 启动时从上次的 hand_eye.npz 载入 joints, 每按一次 r 出队执行一个
    replay_queue = deque()
    if os.path.exists(HAND_EYE_NPZ):
        try:
            _data = np.load(HAND_EYE_NPZ, allow_pickle=False)
            if "joints_all" in _data.files:
                replay_queue = deque(list(_data["joints_all"]))
        except Exception as e:
            print(f"[提示] 读取 {HAND_EYE_NPZ} 失败: {e}")
    print("实时预览窗口已打开, 按键需在窗口聚焦时生效:")
    print("  c=采集当前位姿  s=求解  r=单步复现下一个历史位姿  q=退出")
    if replay_queue:
        print(f"[提示] 已载入 {len(replay_queue)} 组历史 joints ({HAND_EYE_NPZ}), 按 [r] 逐个执行")
    if os.path.exists(CRASH_DUMP):
        print(f"[提示] 检测到崩溃恢复数据: {CRASH_DUMP}")
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            # ---- 实时取帧 + 显示 (主循环不阻塞) ----
            bgr = grab_color(pl)
            if bgr is None:
                continue
            disp = bgr.copy()
            cv2.putText(disp, f"samples={len(samples)}  [c]collect [s]solve [q]quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(WIN_NAME, disp)
            # 每帧 spin 一下, 让 ROS 订阅回调持续触发 (刷新 pose/joints 缓存, 否则锁死在初值)
            arm.spin_once(0.001)
            # waitKey 每轮返回一个按键 (下一轮自动覆盖, 无需全局变量)
            k = cv2.waitKey(1) & 0xFF
            ch = chr(k).lower()
            if ch == "q":
                exited_normally = True
                break
            elif ch == "c":                               # ---- 采集: 记录数据并推入数组 ----
                idx = collect_at_current_pose(arm, pl, samples, idx)
            elif ch == "s":                               # ---- 求解 ----
                if len(samples) < MIN_SAMPLES:
                    print(f"  至少 {MIN_SAMPLES} 组"); continue
                X, (n, rmean, rmax, tmean, tmax) = solve_hand_eye(samples, K, dist)
                print("\nX = T_cam2gripper (mm) =\n",
                      np.array2string(X, precision=4, suppress_small=True))
                print(f"残差({n}帧): 旋转 {rmean:.3f}/{rmax:.3f}°(均/最大)  "
                      f"平移 {tmean:.2f}/{tmax:.2f}mm(均/最大)")
                # 存盘: 所有平移统一为 mm (与 X、标定板一致); joints 为弧度
                T_g2b_mm = np.stack([s["T_gripper2base"].copy() for s in samples])
                T_g2b_mm[:, :3, 3] *= 1000.0          # FK 米 -> mm
                np.savez(f"{OUT_DIR}/hand_eye.npz", X=X, K=K, dist=dist,
                         T_gripper2base_all_mm=T_g2b_mm,
                         joints_all=np.array([s["joints"] for s in samples]),
                         units="translation in mm, joints in rad")
                print(f"  -> {OUT_DIR}/hand_eye.npz  (平移单位: mm)")
            elif ch == "r":                               # ---- 单步复现: 出队一个 joints 执行 ----
                if not replay_queue:
                    print("  复现队列已空 (无历史 joints, 或已全部执行完)")
                    continue
                j = replay_queue.popleft()                # 从头部取出一个 (出队)
                remain = len(replay_queue)
                print(f"  [复现] 执行第 {idx+1} 个位姿 (队列剩 {remain} 个)")
                print(f"         joints={np.round(j, 4).tolist()}")
                try:
                    arm.move_joint(list(j), block=True, timeout=30.0)
                except Exception as e:
                    print(f"  [复现] move_joint 失败: {e}"); continue
                time.sleep(0.5)                            # 等停稳 + 位姿刷新
                idx = collect_at_current_pose(arm, pl, samples, idx)
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        import traceback
        print(f"\n[崩溃] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # 兜底: 非正常退出(崩溃/中断)时存档, 避免下次手调。正常 q 退出不存。
        if not exited_normally:
            dump_samples_on_crash(samples)
        cv2.destroyAllWindows()
        pl.stop(); arm.close(); shutdown()


if __name__ == "__main__":
    main()
