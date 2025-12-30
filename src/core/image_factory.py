import numpy as np
import cv2
from scipy.ndimage import uniform_filter

from core.config import (
    TEXTURE_TYPES, BACKGROUND_MODES, OBJECT_TYPES, SHAPE_WEIGHTS,
    IMAGE_SIZE, SUPERSAMPLE_FACTOR
)
# ==========================================
# 1. UTILITIES
# ==========================================
class Utils:
    """Helper functions for geometry and randomness."""
    
    @staticmethod
    def rotate_coords(u, v, angle):
        """Rotates coordinates (u, v) by a given angle."""
        c, s = np.cos(angle), np.sin(angle)
        return c*u + s*v, -s*u + c*v

    @staticmethod
    def sample_size(max_dim, frac_min=0.05, frac_max=0.6, min_px=16.0):
        """Samples a random size for a shape based on image dimensions."""
        val = np.random.uniform(frac_min, frac_max) * max_dim
        return max(min_px, val)

    @staticmethod
    def polygon_mask(vertices, X, Y):
        """Creates a boolean mask for a polygon defined by vertices."""
        mask = np.ones_like(X, dtype=bool)
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            ex, ey = x2 - x1, y2 - y1
            mask &= ((X - x1) * ey - (Y - y1) * ex >= 0)
        return mask

    @staticmethod
    def random_gray_pair(min_diff=0.25):
        """Returns two distinct gray levels."""
        r1 = np.random.rand()
        x = 1 - 2 * min_diff
        g1 = min_diff + x * r1
        g2 = np.random.choice([0, 1 - x*np.random.rand()]) + g1 - min_diff
        return g1, g2

# ==========================================
# 2. TEXTURE ENGINE
# ==========================================
class TextureEngine:
    """Handles pattern generation and blending logic."""
    
    @staticmethod
    def create(mask, u, v=None, texture_type=None):
        if not np.any(mask):
            return np.zeros_like(mask, dtype=np.float32)

        if texture_type is None:
            texture_type = np.random.choice(TEXTURE_TYPES)

        out = np.zeros_like(mask, dtype=np.float32)

        # Dispatch texture generation
        if texture_type == "white":
            out[mask] = 1.0
        elif texture_type == "gray":
            out[mask] = np.random.uniform(0.1, 0.9)
        elif "stripes" in texture_type:
            TextureEngine._apply_stripes(out, mask, u, texture_type)
        elif texture_type == "checker":
            TextureEngine._apply_checker(out, mask, u, v)
        elif texture_type == "noise":
            TextureEngine._apply_noise(out, mask)
        elif "grad" in texture_type:
            TextureEngine._apply_gradient(out, mask, u, v, texture_type)
        elif texture_type == "multiband":
            TextureEngine._apply_multiband(out, mask, u, v)
        
        return out

    @staticmethod
    def blend(target_slice, new_vals, mask):
        """Blends new values into target slice using random alpha or replacement."""
        if not np.any(mask): return target_slice
        
        # 50% Hard replace, 50% Alpha blend
        if np.random.rand() < 0.5:
            target_slice[mask] = new_vals[mask]
        else:
            alpha = np.random.uniform(0.4, 0.8)
            target_slice[mask] = (1 - alpha) * target_slice[mask] + alpha * new_vals[mask]
        return target_slice

    # --- Internal Texture Generators ---
    @staticmethod
    def _apply_stripes(out, mask, u, t_type):
        n = 3 if t_type == "stripes3" else 2
        w = np.random.uniform(4.0, 16.0)
        offset = np.random.uniform(-w, w)
        idx = np.floor((u + offset) / w).astype(int)
        gs = np.random.rand(n).astype(np.float32)
        while max(gs) - min(gs) < 0.3: gs = np.random.rand(n).astype(np.float32)
        out[mask] = gs[np.mod(idx, n)][mask]

    @staticmethod
    def _apply_checker(out, mask, u, v):
        v_use = v if v is not None else np.zeros_like(u)
        cell = np.random.uniform(6.0, 18.0)
        k = np.random.randint(2, 5)
        idx = np.floor(u / cell).astype(int) + np.floor(v_use / cell).astype(int)
        gs = np.random.rand(k).astype(np.float32)
        while max(gs) - min(gs) < 0.3: gs = np.random.rand(k).astype(np.float32)
        out[mask] = gs[np.mod(idx, k)][mask]

    @staticmethod
    def _apply_noise(out, mask):
        base = np.random.uniform(0.2, 0.8)
        amp = np.random.uniform(0.05, 0.25)
        noise = base + amp * (2 * np.random.rand(*mask.shape) - 1.0)
        out[mask] = np.clip(noise, 0.0, 1.0)[mask]

    @staticmethod
    def _apply_gradient(out, mask, u, v, t_type):
        metric = u if t_type == "grad_lr" else -np.sqrt(u**2 + (v if v is not None else 0)**2)
        mn, mx = metric.min(), metric.max()
        norm = (metric - mn) / (mx - mn + 1e-6)
        base = np.random.uniform(0.1, 0.8)
        out[mask] = np.clip(base + (1.0 - base) * norm, 0.0, 1.0)[mask]

    @staticmethod
    def _apply_multiband(out, mask, u, v):
        coord = np.sqrt(u**2 + v**2) if v is not None else np.abs(u)
        coord -= coord.min()
        bands = np.random.randint(2, 5)
        w = (coord.max() + 1e-6) / bands
        idx = np.clip(np.floor(coord / (w + 1e-6)).astype(int), 0, bands - 1)
        gs = np.random.rand(bands).astype(np.float32)
        while max(gs) - min(gs) < 0.3: gs = np.random.rand(bands).astype(np.float32)
        out[mask] = gs[idx][mask]

# ==========================================
# 3. SHAPE DRAWER (REFACTORED)
# ==========================================
class ShapeDrawer:
    def __init__(self, img, X, Y, H, W):
        self.img = img
        self.X = X
        self.Y = Y
        self.H = H
        self.W = W
        self.max_dim = max(H, W)

    # --- Core Helpers for DRY Code ---

    def _get_local_grid(self, cx, cy, radius_or_dims):
        """
        Calculates ROI slices and returns local coordinates (u, v) centered at (cx, cy).
        Returns: sl_y, sl_x, u, v. Returns None if ROI is invalid.
        """
        if isinstance(radius_or_dims, (tuple, list, np.ndarray)):
            rw, rh = radius_or_dims
        else:
            rw = rh = radius_or_dims

        margin = 2
        x_min = int(np.floor(cx - rw - margin))
        x_max = int(np.ceil(cx + rw + margin))
        y_min = int(np.floor(cy - rh - margin))
        y_max = int(np.ceil(cy + rh + margin))

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(self.W, x_max), min(self.H, y_max)

        if x_max <= x_min or y_max <= y_min:
            return None

        sl_y = slice(y_min, y_max)
        sl_x = slice(x_min, x_max)
        
        # Extract global coords for ROI and shift to local
        u = self.X[sl_y, sl_x] - cx
        v = self.Y[sl_y, sl_x] - cy
        
        return sl_y, sl_x, u, v

    def _render_mask(self, mask, u, v, sl_y, sl_x):
        """Applies texture and blends mask into the main image."""
        if mask is None or not np.any(mask):
            return
        
        vals = TextureEngine.create(mask, u, v)
        self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], vals, mask)

    # --- Main Draw Entry Point ---
    
    def draw(self, shape_type):
        # 2D Geometry
        if shape_type in ["circle", "ellipse", "rect"]:
            self._draw_basic_2d(shape_type)
        elif shape_type in ["triangle", "polygon", "ring", "blob", "arc"]:
            self._draw_complex_2d(shape_type)
        elif shape_type == "line":
            self._draw_line_primitive()
            
        # Optical Patterns
        elif shape_type in ["siemens", "bullseye", "slanted_edge"]:
            self._draw_optical(shape_type)
            
        # 3D Objects
        elif shape_type in ["sphere3d", "donut3d", "cube3d", "pyramid3d"]:
            self._draw_3d(shape_type)
            
        # Layouts & Misc
        elif shape_type in ["lamp_row", "window_grid"]:
            self._draw_layout(shape_type)
        elif shape_type == "scribble":
            self._draw_scribble()

    # --- Shape Logic Implementation ---

    def _draw_basic_2d(self, stype):
        # 1. Determine size & position
        if stype == "rect":
            hw, hh = Utils.sample_size(self.max_dim), Utils.sample_size(self.max_dim)
            R_bound = np.sqrt(hw**2 + hh**2)
        elif stype == "ellipse":
            a, b = Utils.sample_size(self.max_dim), Utils.sample_size(self.max_dim)
            R_bound = max(a, b)
        else: # circle
            r = Utils.sample_size(self.max_dim)
            R_bound = r

        cx, cy = np.random.uniform(-R_bound, self.W+R_bound), np.random.uniform(-R_bound, self.H+R_bound)

        # 2. Get Grid
        res = self._get_local_grid(cx, cy, R_bound)
        if not res: return
        sl_y, sl_x, u, v = res

        # 3. Compute Mask
        if stype == "circle":
            mask = (u**2 + v**2) <= R_bound**2
        else:
            # Rotate coords for non-circular shapes
            rot = np.random.uniform(0, 2*np.pi)
            ur, vr = Utils.rotate_coords(u, v, rot)
            
            if stype == "rect":
                mask = (np.abs(ur) <= hw) & (np.abs(vr) <= hh)
            else: # ellipse
                mask = (ur**2/a**2 + vr**2/b**2) <= 1.0

        # 4. Render
        self._render_mask(mask, u, v, sl_y, sl_x)

    def _draw_complex_2d(self, stype):
        R = Utils.sample_size(self.max_dim)
        cx, cy = np.random.uniform(-R, self.W+R), np.random.uniform(-R, self.H+R)
        res = self._get_local_grid(cx, cy, R * (1.3 if stype == "blob" else 1.0))
        if not res: return
        sl_y, sl_x, u, v = res

        mask = None
        if stype == "ring":
            r_in = np.random.uniform(0.3*R, 0.8*R)
            d2 = u**2 + v**2
            mask = (d2 <= R**2) & (d2 >= r_in**2)
            
        elif stype == "blob":
            r_pol, theta = np.sqrt(u**2+v**2), np.arctan2(v, u)
            amp = np.random.uniform(0.1, 0.3) * R
            k = np.random.randint(2, 7)
            pert = R + amp * np.cos(k * theta + np.random.rand()*6)
            mask = r_pol <= pert
            
        elif stype == "arc":
            r_in = np.random.uniform(0.3*R, 0.8*R)
            r_pol = np.sqrt(u**2+v**2)
            theta = np.mod(np.arctan2(v, u), 2*np.pi)
            start = np.random.uniform(0, 6.28)
            span = np.random.uniform(0.5, 4.5)
            theta_shifted = np.mod(theta - start, 2*np.pi)
            mask = (r_pol >= r_in) & (r_pol <= R) & (theta_shifted <= span)

        elif stype in ["triangle", "polygon"]:
            ur, vr = Utils.rotate_coords(u, v, np.random.uniform(0, 6.28))
            if stype == "triangle":
                s = Utils.sample_size(self.max_dim)
                h = np.sqrt(3)/2*s
                verts = [np.array([0, -2*h/3]), np.array([-s/2, h/3]), np.array([s/2, h/3])]
                mask = Utils.polygon_mask(verts, ur, vr)
            else:
                ns = np.random.randint(3, 9)
                angs = np.linspace(0, 6.28, ns, endpoint=False)
                verts = np.stack([R*np.cos(angs), R*np.sin(angs)], axis=1)
                mask = Utils.polygon_mask(verts, ur, vr)

        self._render_mask(mask, u, v, sl_y, sl_x)

    def _draw_optical(self, stype):
        R = Utils.sample_size(self.max_dim)
        cx, cy = np.random.uniform(-R, self.W+R), np.random.uniform(-R, self.H+R)
        res = self._get_local_grid(cx, cy, R)
        if not res: return
        sl_y, sl_x, u, v = res
        r = np.sqrt(u**2 + v**2)
        base_mask = r <= R

        if stype == "siemens":
            if np.any(base_mask):
                th = np.arctan2(v, u) + np.random.rand()*6
                n = np.random.randint(12, 32)
                sec = np.floor((th + np.pi) / (2*np.pi/n)).astype(int)
                vals = np.where(sec % 2 == 0, 0.1, 0.9).astype(np.float32)
                self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], vals, base_mask)

        elif stype == "bullseye":
            if np.any(base_mask):
                n = np.random.randint(4, 12)
                idx = np.floor(r / (R/n)).astype(int)
                vals = np.where(idx % 2 == 0, 0.2, 0.8).astype(np.float32)
                self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], vals, base_mask)
        
        elif stype == "slanted_edge":
            ang = np.random.choice([0, 1.57]) + np.random.uniform(-0.2, 0.2)
            nx, ny = np.cos(ang), np.sin(ang)
            d = (self.X - cx)*nx + (self.Y - cy)*ny
            vals = np.where(d < 0, 0.2, 0.8).astype(np.float32)
            band = np.abs(d) < 2
            vals[band] = 0.2*(1 - (d[band]/4+0.5)) + 0.8*(d[band]/4+0.5)
            self.img = TextureEngine.blend(self.img, vals, np.ones_like(self.img, dtype=bool))

    def _draw_3d(self, stype):
        if stype in ["cube3d", "pyramid3d"]:
            L = Utils.sample_size(min(self.H, self.W))
            R_safe = L * 1.5
            cx, cy = np.random.uniform(-R_safe, self.W), np.random.uniform(-R_safe, self.H)
            res = self._get_local_grid(cx, cy, R_safe)
            if not res: return
            sl_y, sl_x, u, v = res
            ang = np.random.rand() * 6.28
            c, s = np.cos(ang), np.sin(ang)
            Rm = np.array([[c, -s], [s, c]], dtype=np.float32)
            
            # Local vertex transformer
            def tr(p):
                pr = Rm @ p
                # Returns local coord relative to center, not global
                return np.array([pr[0], pr[1]]) 

            # Define vertices relative to center
            b0, b1 = tr(np.array([-0.5, -0.5])*L), tr(np.array([0.5, -0.5])*L)
            b2, b3 = tr(np.array([0.5, 0.5])*L), tr(np.array([-0.5, 0.5])*L)
            
            faces = []
            base_col = np.random.uniform(0.2, 0.7)
            
            if stype == "cube3d":
                d = Rm @ (np.array([0.4, -0.6])*L) # Depth vector
                faces = [
                    ([b3+d, b2+d, b2, b3], base_col+0.2), 
                    ([b1+d, b2+d, b2, b1], base_col-0.1), 
                    ([b0, b1, b2, b3], base_col+0.1)
                ]
            else: # pyramid
                apex = tr(np.array([0.0, -1.0])*L) 
                faces = [
                    ([b0, b1, apex], base_col+0.2), 
                    ([b1, b2, apex], base_col), 
                    ([b2, b3, apex], base_col-0.1)
                ]

            for verts, col in faces:
                mask = Utils.polygon_mask(verts, u, v)
                if np.any(mask):
                    v_arr = np.full_like(u, col, dtype=np.float32)
                    self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], v_arr, mask)
            return

        # Sphere/Donut
        R = Utils.sample_size(self.max_dim)
        R_bound = R if stype == "sphere3d" else R * 1.35 
        cx, cy = np.random.uniform(-R_bound, self.W+R_bound), np.random.uniform(-R_bound, self.H+R_bound)
        res = self._get_local_grid(cx, cy, R_bound)
        if not res: return
        sl_y, sl_x, u, v = res

        mask = None
        vals = None

        if stype == "sphere3d":
            rsq = (u/R)**2 + (v/R)**2
            mask = rsq <= 1.0
            if np.any(mask):
                z = np.sqrt(np.clip(1.0 - rsq, 0, 1))
                l = np.array([0.5, 0.5, 0.7]); l /= np.linalg.norm(l)
                dot = np.clip((u/R)*l[0] + (v/R)*l[1] + z*l[2], 0, 1)
                vals = np.clip(0.2 + 0.8*dot, 0, 1)

        elif stype == "donut3d":
            rt = np.random.uniform(0.1, 0.35) * R
            r_xy = np.sqrt(u**2 + v**2) + 1e-6
            t = (r_xy - R) / rt
            mask = np.abs(t) <= 1.0
            if np.any(mask):
                nrx, nry = u/r_xy, v/r_xy
                l = np.array([0.6, 0.4]); l /= np.linalg.norm(l)
                d = np.clip(nrx*l[0] + nry*l[1], 0, 1)
                tube = np.clip(1 - 0.5*np.abs(t), 0, 1)
                vals = np.clip(0.2 + 0.7*d*tube, 0, 1)

        if mask is not None and vals is not None:
             self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], vals, mask)

    def _draw_layout(self, stype):
        if stype == "lamp_row":
            n = np.random.randint(4, 10)
            length = np.random.uniform(0.4, 0.9)*self.max_dim
            r_lamp = Utils.sample_size(self.max_dim)*0.25
            cx, cy = np.random.rand(2)*[self.W, self.H]
            ang = np.random.rand()*3.14
            dx, dy = np.cos(ang), np.sin(ang)
            sx, sy = cx-0.5*length*dx, cy-0.5*length*dy
            base = np.random.uniform(0.5, 0.9)
            
            for i in range(n):
                t = (i+0.5)/n
                lx, ly = sx+t*length*dx, sy+t*length*dy
                res = self._get_local_grid(lx, ly, r_lamp)
                if not res: continue
                sl_y, sl_x, u, v = res
                
                mask = (u**2 + v**2) <= r_lamp**2
                if np.any(mask):
                    r_norm = np.sqrt(u**2 + v**2) / (r_lamp + 1e-6)
                    vals = np.clip(base + (1-base)*(1-r_norm), 0, 1)
                    self.img[sl_y, sl_x] = TextureEngine.blend(self.img[sl_y, sl_x], vals, mask)

        elif stype == "window_grid":
            rows, cols = np.random.randint(3,8), np.random.randint(3,10)
            tw, th = np.random.uniform(0.3, 0.9)*self.W, np.random.uniform(0.3, 0.9)*self.H
            cw, ch = tw/cols, th/rows
            mx, my = np.random.uniform(0, self.W-tw), np.random.uniform(0, self.H-th)
            
            for i in range(rows):
                for j in range(cols):
                    shk = np.random.uniform(0.1, 0.3)
                    x0 = mx + j*cw + cw*shk
                    y0 = my + i*ch + ch*shk
                    w, h = cw*(1-shk), ch*(1-shk)
                    cx, cy = x0 + w/2, y0 + h/2
                    res = self._get_local_grid(cx, cy, (w/2, h/2))
                    if not res: continue
                    sl_y, sl_x, u, v = res
                    
                    mask = (np.abs(u) <= w/2) & (np.abs(v) <= h/2)
                    self._render_mask(mask, u, v, sl_y, sl_x)

    def _draw_line_primitive(self):
        l = Utils.sample_size(self.max_dim)*2
        cx, cy = np.random.rand(2)*[self.W, self.H]
        ang = np.random.rand()*6.28
        vx, vy = np.cos(ang)*l/2, np.sin(ang)*l/2
        steps = int(l)
        x = np.linspace(cx-vx, cx+vx, steps).astype(int)
        y = np.linspace(cy-vy, cy+vy, steps).astype(int)
        valid = (x>=0)&(x<self.W)&(y>=0)&(y<self.H)
        self.img[y[valid], x[valid]] = np.random.rand()

    def _draw_scribble(self):
        thick = np.random.uniform(2.0, 5.0)
        subtype = np.random.choice(["random", "spiral", "loops", "graffiti"], p=[0.4, 0.25, 0.2, 0.15])
        
        pts = None
        if subtype == "random": pts = self._gen_random_polyline((40, 120))
        elif subtype == "spiral": pts = self._gen_spiral_polyline()
        elif subtype == "loops": pts = self._gen_loop_polyline()
        else:
            parts = []
            for _ in range(np.random.randint(2,5)):
                c = np.random.choice(["r", "s", "l"], p=[0.5, 0.25, 0.25])
                if c=="r": parts.append(self._gen_random_polyline((20,70)))
                elif c=="s": parts.append(self._gen_spiral_polyline())
                else: parts.append(self._gen_loop_polyline())
            pts = np.concatenate(parts, axis=0)
            
        mask = self._rasterize_polyline(pts, thick)
        if np.any(mask):
            vals = TextureEngine.create(mask, self.X, self.Y)
            self.img = TextureEngine.blend(self.img, vals, mask)

    # --- Polyline Generators (Internal) ---
    def _gen_random_polyline(self, n_range):
        n = np.random.randint(*n_range)
        step = 0.04 * min(self.H, self.W)
        x, y = np.zeros(n), np.zeros(n)
        x[0], y[0] = np.random.rand(2)*[self.W, self.H]
        for i in range(1, n):
            x[i] = np.clip(x[i-1]+np.random.randn()*step, 0, self.W)
            y[i] = np.clip(y[i-1]+np.random.randn()*step, 0, self.H)
        return np.stack([x, y], axis=1)

    def _gen_spiral_polyline(self):
        cx, cy = np.random.uniform(0.3*self.W, 0.7*self.W), np.random.uniform(0.3*self.H, 0.7*self.H)
        turns = np.random.uniform(2.5, 5.5)
        max_r = np.random.uniform(0.2, 0.45)*min(self.H, self.W)
        t = np.linspace(0, 2*np.pi*turns, 400)
        r = (t/(2*np.pi*turns))*max_r + 0.1*max_r*np.sin(5*t)
        return np.stack([cx+r*np.cos(t), cy+r*np.sin(t)], axis=1)

    def _gen_loop_polyline(self):
        cx, cy = np.random.uniform(0.2*self.W, 0.8*self.W), np.random.uniform(0.2*self.H, 0.8*self.H)
        t = np.linspace(0, 2*np.pi*np.random.randint(3,7), 500)
        ax, ay = np.random.uniform(0.15,0.3)*self.W, np.random.uniform(0.15,0.3)*self.H
        x = cx + ax*np.sin(t*np.random.uniform(1,2))
        y = cy + ay*np.sin(t*np.random.uniform(2,4) + np.random.rand())
        return np.stack([x, y], axis=1)

    def _rasterize_polyline(self, points, thickness, factor=3):
        Hs, Ws = (self.H + factor - 1) // factor, (self.W + factor - 1) // factor
        xs = (np.arange(Ws) + 0.5) * (self.W / Ws)
        ys = (np.arange(Hs) + 0.5) * (self.H / Hs)
        Xs, Ys = np.meshgrid(xs, ys)
        mask_s = np.zeros((Hs, Ws), dtype=bool)
        if points.shape[0] < 2: return np.zeros((self.H, self.W), dtype=bool)
        r2 = (thickness/2)**2
        for i in range(len(points)-1):
            x0, y0 = points[i]; x1, y1 = points[i+1]
            vx, vy = x1-x0, y1-y0
            l2 = vx*vx+vy*vy+1e-8
            px, py = Xs-x0, Ys-y0
            t = np.clip((px*vx+py*vy)/l2, 0, 1)
            dist2 = (Xs-(x0+t*vx))**2 + (Ys-(y0+t*vy))**2
            mask_s |= (dist2 <= r2)
        return np.kron(mask_s, np.ones((factor, factor), dtype=bool))[:self.H, :self.W]

# ==========================================
# 4. IMAGE GENERATOR (PUBLIC API)
# ==========================================
class ImageGenerator:
    @staticmethod
    def generate_background(H, W, X, Y):
        mode = np.random.choice(BACKGROUND_MODES)
        
        if mode == "constant":
            return np.full((H, W), np.random.rand(), dtype=np.float32)
        elif mode == "black":
            return np.zeros((H, W), dtype=np.float32)
        elif mode == "white":
            return np.ones((H, W), dtype=np.float32)
        elif mode == "half":
            v1, v2 = Utils.random_gray_pair()
            cx, cy = W/2.0, H/2.0
            ang = np.random.rand()*np.pi
            dist = (X - cx)*np.cos(ang) + (Y - cy)*np.sin(ang)
            return np.where(dist >= 0, v1, v2).astype(np.float32)
        elif mode == "stripes":
            img = np.zeros((H, W), dtype=np.float32)
            TextureEngine._apply_stripes(img, np.ones((H, W), bool), X, "stripes3")
            return img
        elif mode == "checker":
            img = np.zeros((H, W), dtype=np.float32)
            TextureEngine._apply_checker(img, np.ones((H, W), bool), X, Y)
            return img
        
        return np.zeros((H, W), dtype=np.float32)

    @staticmethod
    def create_clean_image(max_tries=6):
        H_tgt, W_tgt = IMAGE_SIZE, IMAGE_SIZE
        factor = SUPERSAMPLE_FACTOR
        
        H_high = int(H_tgt * factor)
        W_high = int(W_tgt * factor)
        
        for _ in range(max_tries):
            X_high, Y_high = np.meshgrid(np.arange(W_high, dtype=np.float32), 
                                         np.arange(H_high, dtype=np.float32))
            
            img_high = ImageGenerator.generate_background(H_high, W_high, X_high, Y_high)
            drawer = ShapeDrawer(img_high, X_high, Y_high, H_high, W_high)
            
            n_objects = np.random.randint(6, 40)
            keys = list(SHAPE_WEIGHTS.keys())
            probs = np.array(list(SHAPE_WEIGHTS.values()))
            probs /= probs.sum()
            
            chosen_shapes = []
            for _ in range(n_objects):
                if np.random.rand() < 0.6:
                     chosen_shapes.append(np.random.choice(keys, p=probs))
                else:
                     chosen_shapes.append(np.random.choice(OBJECT_TYPES))

            for stype in chosen_shapes:
                drawer.draw(stype)
            
            # Downsample
            img_clean = np.clip(drawer.img if factor == 1.0 else cv2.resize(drawer.img, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA), 0.0, 1.0).astype(np.float32)
            if ImageGenerator._is_quality_ok(img_clean):
                return img_clean
                
        return img_clean

    @staticmethod
    def _is_quality_ok(img):
        return img.std() >= 0.05