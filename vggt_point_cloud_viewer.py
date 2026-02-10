import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT on an image and view the point cloud in OpenGL."
    )
    parser.add_argument("image", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--preprocess-mode",
        type=str,
        default="crop",
        choices=["crop", "pad"],
        help="Preprocessing mode before VGGT inference (default: crop).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Force a device selection; defaults to cuda if available.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=1.01,
        help="Filter depth points with confidence below this value (default: 1.01).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of points to display (default: 200000).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride when sampling pixels (default: 1).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="OpenGL point size (default: 1.5).",
    )
    return parser.parse_args()


def load_model(device: torch.device) -> VGGT:
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def load_image_for_colors(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.float32) / 255.0


def run_vggt(image_path: Path, device: torch.device, preprocess_mode: str):
    images = load_and_preprocess_images([str(image_path)], mode=preprocess_mode)
    images = images.to(device)
    model_height, model_width = images.shape[-2:]
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = load_model(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype, enabled=device.type == "cuda"):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    if extrinsic is None or intrinsic is None:
        raise ValueError("Camera predictions are missing from VGGT output.")
    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    if depth is None or depth_conf is None:
        raise ValueError("Depth predictions are missing from VGGT output.")

    depth = depth.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    colors = images.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return depth, depth_conf, extrinsic, intrinsic, colors, (model_width, model_height)


def build_point_cloud(
    depth: np.ndarray,
    depth_conf: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    colors: np.ndarray,
    conf_threshold: float,
    stride: int,
    max_points: int,
):
    if depth.ndim == 4 and depth.shape[0] == 1 and depth.shape[-1] == 1:
        depth_2d = depth[0, :, :, 0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth_2d = depth[:, :, 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth_2d = depth[0]
    elif depth.ndim == 2:
        depth_2d = depth
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    depth_map = depth_2d[:, :, None]

    if depth_conf.ndim == 4 and depth_conf.shape[0] == 1 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[0, :, :, 0]
    elif depth_conf.ndim == 3 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[:, :, 0]
    elif depth_conf.ndim == 3 and depth_conf.shape[0] == 1:
        depth_conf = depth_conf[0]

    if stride > 1:
        depth_map = depth_map[::stride, ::stride]
        depth_conf = depth_conf[::stride, ::stride]
        colors = colors[::stride, ::stride]

    if extrinsic.ndim == 3 and extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    if intrinsic.ndim == 3 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]

    world_points = unproject_depth_map_to_point_map(
        depth_map[None, ...], extrinsic[None, ...], intrinsic[None, ...]
    )[0]

    conversion = np.array([1.0, -1.0, -1.0], dtype=np.float32)
    world_points = world_points * conversion

    valid_mask = (depth_map.squeeze(-1) > 1e-6) & (depth_conf >= conf_threshold)
    points = world_points[valid_mask]
    point_colors = colors[valid_mask]
    conf_values = depth_conf[valid_mask]

    if max_points > 0 and points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]
        point_colors = point_colors[indices]
        conf_values = conf_values[indices]

    return points.astype(np.float32), point_colors.astype(np.float32), conf_values


def compute_depth_colors(points: np.ndarray) -> np.ndarray:
    depth = points[:, 2]
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    depth_min = float(np.percentile(depth, 5))
    depth_max = float(np.percentile(depth, 95))
    span = max(depth_max - depth_min, 1e-6)
    t = np.clip((depth - depth_min) / span, 0.0, 1.0)
    return np.stack([t, 1.0 - t, np.full_like(t, 0.1)], axis=-1).astype(np.float32)


def scale_intrinsic(
    intrinsic: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray:
    if intrinsic.ndim > 2:
        intrinsic = intrinsic.squeeze()
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic 3x3, got {intrinsic.shape}")
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / max(src_w, 1)
    scale_y = dst_h / max(src_h, 1)
    scaled = np.array(intrinsic, copy=True)
    scaled[0, 0] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[0, 2] *= scale_x
    scaled[1, 2] *= scale_y
    return scaled


def projection_from_intrinsic(
    intrinsic: np.ndarray, width: int, height: int, near: float, far: float
) -> np.ndarray:
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 2.0 * cx / width - 1.0
    proj[1, 2] = 1.0 - 2.0 * cy / height
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    depth, depth_conf, extrinsic, intrinsic, colors, model_size = run_vggt(
        args.image, device, args.preprocess_mode
    )

    points, point_colors, conf_values = build_point_cloud(
        depth,
        depth_conf,
        extrinsic,
        intrinsic,
        colors,
        conf_threshold=args.conf_threshold,
        stride=max(args.stride, 1),
        max_points=max(args.max_points, 0),
    )

    if points.size == 0 and args.conf_threshold > 0.0:
        print("No points after confidence filter; retrying with --conf-threshold 0.0")
        points, point_colors, conf_values = build_point_cloud(
            depth,
            depth_conf,
            extrinsic,
            intrinsic,
            colors,
            conf_threshold=0.0,
            stride=max(args.stride, 1),
            max_points=max(args.max_points, 0),
        )

    if points.size == 0:
        raise ValueError(
            "No points to display. Try lowering --stride or check depth output."
        )

    depth_colors = compute_depth_colors(points)

    try:
        import pyglet
        from pyglet import gl
        from pyglet.graphics import shader
    except ImportError as exc:
        raise ImportError(
            "pyglet is required for viewing. Install with: pip install pyglet"
        ) from exc

    def translate_matrix(x: float, y: float, z: float) -> np.ndarray:
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return mat

    def rotate_x_matrix(deg: float) -> np.ndarray:
        rad = math.radians(deg)
        c = math.cos(rad)
        s = math.sin(rad)
        mat = np.eye(4, dtype=np.float32)
        mat[1, 1] = c
        mat[1, 2] = -s
        mat[2, 1] = s
        mat[2, 2] = c
        return mat

    def rotate_y_matrix(deg: float) -> np.ndarray:
        rad = math.radians(deg)
        c = math.cos(rad)
        s = math.sin(rad)
        mat = np.eye(4, dtype=np.float32)
        mat[0, 0] = c
        mat[0, 2] = s
        mat[2, 0] = -s
        mat[2, 2] = c
        return mat

    def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward = forward / max(np.linalg.norm(forward), 1e-6)
        right = np.cross(forward, up)
        right = right / max(np.linalg.norm(right), 1e-6)
        true_up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(true_up, eye)
        view[2, 3] = np.dot(forward, eye)
        return view

    class PointCloudViewer(pyglet.window.Window):
        def __init__(
            self,
            vertices,
            rgb_colors,
            depth_colors,
            point_size,
            intrinsic,
            model_size,
            extrinsic,
        ):
            super().__init__(
                width=1200,
                height=800,
                caption="VGGT Point Cloud Viewer",
                resizable=True,
            )
            self.vertices = vertices
            self.rgb_colors = rgb_colors
            self.depth_colors = depth_colors
            self.use_depth_colors = False
            self.point_size = point_size
            self.vertex_list = None
            self.program = None
            self.intrinsic = intrinsic
            self.model_size = model_size
            self.extrinsic = extrinsic

            self.center = self.vertices.mean(axis=0)
            radius = np.linalg.norm(self.vertices - self.center, axis=1).max()
            self.distance = max(radius * 2.5, 0.1)
            self.yaw = 0.0
            self.pitch = 0.0
            self.camera_pos = self.center + np.array(
                [0.0, 0.0, self.distance], dtype=np.float32
            )
            self.mouse_look = False
            self.move_speed = max(radius * 0.5, 0.05)
            self.look_sensitivity = 0.15
            self.use_vggt_pose = True
            self.move_state = {
                "forward": False,
                "backward": False,
                "left": False,
                "right": False,
                "up": False,
                "down": False,
                "boost": False,
            }

            self._reset_camera()
            self._init_gl()
            self._update_vertex_list()
            pyglet.clock.schedule_interval(self._update_camera, 1.0 / 60.0)

        def _reset_camera(self):
            if self.use_vggt_pose and self.extrinsic is not None:
                self._init_camera_from_extrinsic(self.extrinsic)
            else:
                self.yaw = 0.0
                self.pitch = 0.0
                self.camera_pos = self.center + np.array(
                    [0.0, 0.0, self.distance], dtype=np.float32
                )

        def _init_camera_from_extrinsic(self, extrinsic):
            if extrinsic is None:
                return
            if extrinsic.ndim == 3 and extrinsic.shape[0] == 1:
                extr = extrinsic[0]
            else:
                extr = extrinsic
            if extr.shape == (3, 4):
                extr_4 = np.eye(4, dtype=np.float32)
                extr_4[:3, :] = extr
            else:
                extr_4 = extr.astype(np.float32)

            r = extr_4[:3, :3]
            t = extr_4[:3, 3]
            cam_to_world_r = r.T
            cam_to_world_t = -r.T @ t

            conversion = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            cam_to_world_r = conversion @ cam_to_world_r @ conversion
            cam_to_world_t = conversion @ cam_to_world_t

            forward = cam_to_world_r @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
            yaw = math.degrees(math.atan2(forward[0], -forward[2]))
            pitch = math.degrees(math.asin(np.clip(forward[1], -1.0, 1.0)))
            self.yaw = yaw
            self.pitch = pitch
            self.camera_pos = cam_to_world_t.astype(np.float32)

        def _init_gl(self):
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
            gl.glClearColor(0.05, 0.06, 0.07, 1.0)

            vertex_source = """
            #version 330
            in vec3 position;
            in vec3 color;
            uniform mat4 mvp;
            uniform float point_size;
            out vec3 v_color;
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
                gl_PointSize = point_size;
                v_color = color;
            }
            """
            fragment_source = """
            #version 330
            in vec3 v_color;
            out vec4 out_color;
            void main() {
                out_color = vec4(v_color, 1.0);
            }
            """
            self.program = shader.ShaderProgram(
                shader.Shader(vertex_source, "vertex"),
                shader.Shader(fragment_source, "fragment"),
            )

        def _update_vertex_list(self):
            colors = self.depth_colors if self.use_depth_colors else self.rgb_colors
            if self.vertex_list is None:
                self.vertex_list = self.program.vertex_list(
                    len(self.vertices),
                    gl.GL_POINTS,
                    position=("f", self.vertices.flatten()),
                    color=("f", colors.flatten()),
                )
            else:
                self.vertex_list.color = colors.flatten()

        def on_draw(self):
            self.clear()
            near = 0.01
            far = 10000.0

            if self.intrinsic is None:
                proj = projection_from_intrinsic(
                    np.array(
                        [
                            [self.width, 0.0, self.width * 0.5],
                            [0.0, self.height, self.height * 0.5],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    self.width,
                    self.height,
                    near,
                    far,
                )
            else:
                scaled_intrinsic = scale_intrinsic(
                    self.intrinsic, self.model_size, (self.width, self.height)
                )
                proj = projection_from_intrinsic(
                    scaled_intrinsic, self.width, self.height, near, far
                )

            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)
            forward = np.array(
                [
                    math.cos(pitch_rad) * math.sin(yaw_rad),
                    math.sin(pitch_rad),
                    -math.cos(pitch_rad) * math.cos(yaw_rad),
                ],
                dtype=np.float32,
            )
            target = self.camera_pos + forward
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            view = look_at(self.camera_pos, target, up)
            mvp = proj @ view

            self.program.use()
            self.program["mvp"] = mvp.T.astype(np.float32).flatten().tolist()
            self.program["point_size"] = float(self.point_size)
            self.vertex_list.draw(gl.GL_POINTS)

        def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
            if self.mouse_look:
                self.yaw += dx * self.look_sensitivity
                self.pitch += dy * self.look_sensitivity
                self.pitch = max(min(self.pitch, 89.0), -89.0)

        def on_mouse_press(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.mouse_look = True

        def on_mouse_release(self, x, y, button, modifiers):
            if button == pyglet.window.mouse.LEFT:
                self.mouse_look = False

        def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
            self.move_speed *= math.pow(1.1, scroll_y)

        def on_key_press(self, symbol, modifiers):
            if symbol == pyglet.window.key.R:
                self._reset_camera()
            elif symbol == pyglet.window.key.C:
                self.use_depth_colors = not self.use_depth_colors
                self._update_vertex_list()
            elif symbol == pyglet.window.key.T:
                self.use_vggt_pose = not self.use_vggt_pose
                self._reset_camera()
            elif symbol == pyglet.window.key.F:
                self.yaw = 0.0
                self.pitch = 0.0
                self.camera_pos = self.center + np.array(
                    [0.0, 0.0, self.distance], dtype=np.float32
                )
            elif symbol == pyglet.window.key.EQUAL or symbol == pyglet.window.key.PLUS:
                self.point_size = min(self.point_size + 0.5, 10.0)
            elif symbol == pyglet.window.key.MINUS:
                self.point_size = max(self.point_size - 0.5, 0.5)
            elif symbol == pyglet.window.key.W:
                self.move_state["forward"] = True
            elif symbol == pyglet.window.key.S:
                self.move_state["backward"] = True
            elif symbol == pyglet.window.key.A:
                self.move_state["left"] = True
            elif symbol == pyglet.window.key.D:
                self.move_state["right"] = True
            elif symbol == pyglet.window.key.Q:
                self.move_state["down"] = True
            elif symbol == pyglet.window.key.E:
                self.move_state["up"] = True
            elif symbol == pyglet.window.key.LSHIFT:
                self.move_state["boost"] = True

        def on_key_release(self, symbol, modifiers):
            if symbol == pyglet.window.key.W:
                self.move_state["forward"] = False
            elif symbol == pyglet.window.key.S:
                self.move_state["backward"] = False
            elif symbol == pyglet.window.key.A:
                self.move_state["left"] = False
            elif symbol == pyglet.window.key.D:
                self.move_state["right"] = False
            elif symbol == pyglet.window.key.Q:
                self.move_state["down"] = False
            elif symbol == pyglet.window.key.E:
                self.move_state["up"] = False
            elif symbol == pyglet.window.key.LSHIFT:
                self.move_state["boost"] = False

        def _update_camera(self, dt):
            yaw_rad = math.radians(self.yaw)
            pitch_rad = math.radians(self.pitch)
            forward = np.array(
                [
                    math.cos(pitch_rad) * math.sin(yaw_rad),
                    math.sin(pitch_rad),
                    -math.cos(pitch_rad) * math.cos(yaw_rad),
                ],
                dtype=np.float32,
            )
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            norm_right = np.linalg.norm(right)
            if norm_right > 1e-6:
                right = right / norm_right

            speed = self.move_speed * (3.0 if self.move_state["boost"] else 1.0)
            velocity = np.zeros(3, dtype=np.float32)
            if self.move_state["forward"]:
                velocity += forward
            if self.move_state["backward"]:
                velocity -= forward
            if self.move_state["left"]:
                velocity -= right
            if self.move_state["right"]:
                velocity += right
            if self.move_state["up"]:
                velocity += up
            if self.move_state["down"]:
                velocity -= up

            norm = np.linalg.norm(velocity)
            if norm > 1e-6:
                velocity = velocity / norm
                self.camera_pos += velocity * speed * dt

    print("Controls: left-drag look, scroll adjust speed")
    print("Keys: WASD move, Q/E down/up, Shift boost, C toggle depth colors, R reset")

    viewer = PointCloudViewer(
        points,
        point_colors,
        depth_colors,
        args.point_size,
        intrinsic,
        model_size,
        extrinsic,
    )
    pyglet.app.run()


if __name__ == "__main__":
    main()
