#!/usr/bin/env python3
"""
相机查看 tool —— 实时显示 RGB + 深度图。

用哪个相机由 config.yaml 的 camera 字段决定 (懒加载, 见 tools/_hardware.py)。
RGBD 相机 (实现了 get_rgbd) 同时显示彩色 + 深度伪彩; 纯 RGB 相机回退到只显彩色。

深度图:
  - JET 伪彩 (近=红 远=蓝), 有效深度范围 --dmin ~ --dmax (米), 超出涂黑
  - 鼠标悬停深度图 -> 标题栏实时显示该点距离 (米)
  - d/D 调小 dmin, f/F 调大 dmax (步长 0.05m), 便于对不同场景调对比度

运行:
    python main.py camera                  # 经顶层入口
    python -m tools.camera_viewer          # 或直接作为模块
按 q/ESC 退出。
"""
import argparse

import cv2
import numpy as np

from ._hardware import get_camera_class


def render_depth_colormap(depth_mm, dmin_m, dmax_m):
    """uint16 mm 深度图 -> JET 伪彩 BGR。范围 [dmin,dmax] 米映射到全色域, 超出涂黑。
    无效深度 (0) 也涂黑。"""
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = (depth_m >= dmin_m) & (depth_m <= dmax_m)
    vis = np.zeros_like(depth_m, dtype=np.float32)
    vis[valid] = (depth_m[valid] - dmin_m) / max(1e-6, dmax_m - dmin_m)
    vis_u8 = (vis * 255).astype(np.uint8)
    colored = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)
    colored[~valid] = 0   # 无效/超范围涂黑
    return colored


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="相机实时查看 (RGB + 深度)")
    ap.add_argument("--win", default="camera viewer (q/ESC quit)")
    ap.add_argument("--dmin", type=float, default=0.3, help="深度伪彩最近距离 (米)")
    ap.add_argument("--dmax", type=float, default=1.5, help="深度伪彩最远距离 (米)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    CameraImpl = get_camera_class()
    if CameraImpl is None:
        raise RuntimeError("config.yaml 未配置 camera (或为 null), 无法启动相机查看")

    use_rgbd = hasattr(CameraImpl, "get_rgbd")
    # 基类声明了 get_rgbd 但子类可能 raise NotImplementedError -> 运行时探测
    print(f"=== 相机查看: {CameraImpl.__name__} ({'RGBD' if use_rgbd else '仅 RGB'}) ===")
    cam = CameraImpl()
    if hasattr(cam, "K"):
        print("K =\n", cam.K)

    dmin, dmax = args.dmin, args.dmax
    hover_xyz = [None]   # 鼠标 hover 的深度图坐标 (用 list 闭包可变)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            hover_xyz[0] = (x, y)

    win = args.win
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    # 尝试 RGBD 模式; 首次失败则降级到纯 RGB
    rgb_mode = False
    try:
        while True:
            if use_rgbd and not rgb_mode:
                try:
                    bgr, depth_mm = cam.get_rgbd()
                except NotImplementedError:
                    rgb_mode = True
                    cv2.setMouseCallback(win, lambda *a: None)
                    continue
                if bgr is None:
                    continue
                # 对齐 depth 到 bgr 尺寸 (get_rgbd 内部已对齐, 这里保险)
                if depth_mm.shape[:2] != bgr.shape[:2]:
                    depth_mm = cv2.resize(depth_mm, (bgr.shape[1], bgr.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                depth_vis = render_depth_colormap(depth_mm, dmin, dmax)
                # hover 读距 (深度图在右半区: x 减去 RGB 宽度)
                title_extra = ""
                if hover_xyz[0] is not None:
                    hx, hy = hover_xyz[0]
                    dx = hx - bgr.shape[1]   # 深度图区 x
                    if 0 <= hy < depth_mm.shape[0] and 0 <= dx < depth_mm.shape[1]:
                        d = depth_mm[hy, dx]
                        title_extra = (f"  | hover d={d/1000:.3f}m"
                                       if d > 0 else "  | hover d=无效")
                disp = np.hstack([bgr, depth_vis])
                cv2.imshow(win, disp)
                cv2.setWindowTitle(win, f"{win}  [dmin={dmin:.2f} dmax={dmax:.2f}]{title_extra}")
                key = cv2.waitKey(1) & 0xFF
            else:
                bgr = cam.get_frame()
                if bgr is None:
                    continue
                cv2.imshow(win, bgr)
                key = cv2.waitKey(1) & 0xFF

            # d/D/f/F 调深度范围
            if key == ord("d"):
                dmin = max(0.05, dmin - 0.05)
            elif key == ord("D"):
                dmin = min(dmax - 0.05, dmin + 0.05)
            elif key == ord("f"):
                dmax = max(dmin + 0.05, dmax - 0.05)
            elif key == ord("F"):
                dmax = min(10.0, dmax + 0.05)
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        cv2.destroyAllWindows()
        cam.release()


if __name__ == "__main__":
    main()
