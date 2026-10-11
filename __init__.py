# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anupska

import bpy
import bmesh
import gpu
import blf
import math
import mathutils
import mathutils.geometry
import mathutils.bvhtree
import mathutils.kdtree
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils

bl_info = {
    "name": "2D PolySketch Suite",
    "author": "Anupska",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-Panel > Create > 2D PolySketch",
    "description": "Precision CAD 2D sketching, face cutting, and exact miter polyline offset.",
    "category": "Mesh",
}

SNAPPABLE_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT'}

SNAP_RADIUS_PX  = 14.0     # Geometry snap radius
CLOSE_RADIUS_PX = 12.0     # Loop-close radius

SNAP_BIAS_PX = {'VERTEX': 0.0, 'MIDPOINT': 4.0, 'PERPENDICULAR': 6.0, 'EDGE': 10.0}

SNAP_COLORS = {
    'VERTEX': (0.0, 0.85, 1.0, 1.0),
    'MIDPOINT': (1.0, 0.5, 0.0, 1.0),
    'EDGE': (1.0, 0.3, 0.9, 1.0),
    'PERPENDICULAR': (1.0, 0.9, 0.2, 1.0),
}

_active_draw_handlers = []


# ==============================================================================
# 1. SHARED CAD CORE UTILITIES
# ==============================================================================

def is_valid_elem(elem):
    if elem is None:
        return False
    try:
        return elem.is_valid
    except ReferenceError:
        return False

def tag_redraw(context):
    if context and context.area:
        context.area.tag_redraw()

def get_3d_region_and_rv3d(context):
    if context and context.area and context.area.type == 'VIEW_3D':
        for r in context.area.regions:
            if r.type == 'WINDOW':
                space = context.area.spaces.active
                rv3d = space.region_3d if space else None
                if rv3d:
                    return r, rv3d
    return None, None

def get_ui_scale(context):
    try:
        return context.preferences.system.ui_scale
    except Exception:
        return 1.0

def format_unit_length(context, val):
    unit_sys = context.scene.unit_settings.system
    try:
        return bpy.utils.units.to_string(unit_sys, 'LENGTH', val)
    except Exception:
        return f"{val:.3f}m"

def parse_unit_expression(context, s):
    if not s or not s.strip():
        return None
    unit_sys = context.scene.unit_settings.system
    try:
        return float(bpy.utils.units.to_value(unit_sys, 'LENGTH', s))
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def compute_plane_basis(normal):
    n = normal.normalized()
    if abs(n.z) > 0.99:
        u = mathutils.Vector((1.0, 0.0, 0.0))
        v = n.cross(u).normalized()
        u = v.cross(n).normalized()
    else:
        u = mathutils.Vector((0.0, 0.0, 1.0)).cross(n).normalized()
        v = n.cross(u).normalized()
    return u, v

def raycast_plane_intersection(region, rv3d, mouse_2d, plane_origin, plane_normal):
    ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
    ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d).normalized()
    denom = ray_target.dot(plane_normal)
    if abs(denom) > 1e-6:
        t = (plane_origin - ray_origin).dot(plane_normal) / denom
        return ray_origin + t * ray_target
    return ray_origin + ray_target * 5.0

def is_point_in_polygon_2d(pt, poly):
    x, y = pt.x, pt.y
    inside = False
    n = len(poly)
    for i in range(n):
        p1, p2 = poly[i], poly[(i + 1) % n]
        if ((p1.y > y) != (p2.y > y)) and (x < (p2.x - p1.x) * (y - p1.y) / (p2.y - p1.y + 1e-12) + p1.x):
            inside = not inside
    return inside


# ==============================================================================
# 2. NUMPY VECTORIZED GEOMETRY CACHING & SNAPPING
# ==============================================================================

def build_geo_cache(context):
    depsgraph = context.evaluated_depsgraph_get()
    e1_l, e2_l, eobj_l, obj_pts, verts_l = [], [], [], [], []
    
    for inst in depsgraph.object_instances:
        obj = inst.object
        if not inst.show_self or obj.type not in SNAPPABLE_TYPES:
            continue
        try:
            mesh = obj.to_mesh()
        except Exception as ex:
            print(f"[PolySketch Notice] to_mesh conversion warning: {ex}")
            continue
        try:
            nv, ne = len(mesh.vertices), len(mesh.edges)
            if nv == 0:
                continue
            co = np.empty(nv * 3, dtype=np.float32)
            mesh.vertices.foreach_get('co', co)
            mw = np.array(inst.matrix_world, dtype=np.float64)
            wco = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
            ed = np.empty(ne * 2, dtype=np.int32)
            mesh.edges.foreach_get('vertices', ed)
            ed = ed.reshape(-1, 2)
        finally:
            obj.to_mesh_clear()
        idx = len(obj_pts)
        obj_pts.append(wco)
        verts_l.append(wco)
        if ne:
            e1_l.append(wco[ed[:, 0]])
            e2_l.append(wco[ed[:, 1]])
            eobj_l.append(np.full(ne, idx, dtype=np.int32))

    def cat(lst, empty):
        return np.concatenate(lst) if lst else empty

    e1 = cat(e1_l, np.zeros((0, 3)))
    e2 = cat(e2_l, np.zeros((0, 3)))
    return {
        'verts': cat(verts_l, np.zeros((0, 3))),
        'e1': e1, 'e2': e2,
        'mids': (e1 + e2) * 0.5,
        'edge_obj': cat(eobj_l, np.zeros(0, dtype=np.int32)),
        'obj_pts': obj_pts,
        'plane_cache': {},
        'screen_key': None,
    }

def project_np(region, rv3d, pts):
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    m = np.array(rv3d.perspective_matrix, dtype=np.float64)
    h = np.ones((n, 4))
    h[:, :3] = pts
    clip = h @ m.T
    w = clip[:, 3]
    ok = w > 1e-6
    w = np.where(ok, w, 1.0)
    sx = region.width * 0.5 * (1.0 + clip[:, 0] / w)
    sy = region.height * 0.5 * (1.0 + clip[:, 1] / w)
    return np.stack((sx, sy), axis=1), ok

def refresh_screen_cache(cache, region, rv3d):
    key = (tuple(np.array(rv3d.perspective_matrix).ravel()), region.width, region.height)
    if cache['screen_key'] == key:
        return
    cache['screen_key'] = key
    cache['s_verts'], cache['v_ok'] = project_np(region, rv3d, cache['verts'])
    cache['s_mids'],  cache['m_ok'] = project_np(region, rv3d, cache['mids'])
    cache['s_e1'],    cache['e1_ok'] = project_np(region, rv3d, cache['e1'])
    cache['s_e2'],    cache['e2_ok'] = project_np(region, rv3d, cache['e2'])

def nearest_screen(scr, ok, mouse, max_d2):
    if len(scr) == 0:
        return None
    d = scr - mouse
    d2 = np.einsum('ij,ij->i', d, d)
    d2[~ok] = np.inf
    i = int(np.argmin(d2))
    return (i, float(d2[i])) if d2[i] < max_d2 else None

def fit_plane_np(pts):
    if len(pts) < 3:
        return None
    o = pts[0]
    d = pts - o
    far = int(np.argmax(np.einsum('ij,ij->i', d, d)))
    cr = np.cross(d[far], d)
    l2 = np.einsum('ij,ij->i', cr, cr)
    j = int(np.argmax(l2))
    if l2[j] < 1e-12:
        return None
    n = cr[j] / np.sqrt(l2[j])
    return mathutils.Vector(o.tolist()), mathutils.Vector(n.tolist())

def pick_wire_plane(region, rv3d, mouse_2d, view_dir, geo_cache, max_dist=40.0):
    refresh_screen_cache(geo_cache, region, rv3d)
    a, b = geo_cache['s_e1'], geo_cache['s_e2']
    if len(a) == 0:
        return None
    ok = geo_cache['e1_ok'] & geo_cache['e2_ok']
    p = np.array((mouse_2d.x, mouse_2d.y))
    ab = b - a
    l2 = np.einsum('ij,ij->i', ab, ab)
    degenerate = l2 < 1e-9
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.where(degenerate, 1.0, l2), 0.0, 1.0)
    t[degenerate] = 0.0
    diff = (a + ab * t[:, None]) - p
    d2 = np.einsum('ij,ij->i', diff, diff)
    d2[~ok] = np.inf
    i = int(np.argmin(d2))
    if d2[i] > max_dist * max_dist:
        return None
    obj_idx = int(geo_cache['edge_obj'][i])
    pc = geo_cache['plane_cache']
    if obj_idx not in pc:
        pc[obj_idx] = fit_plane_np(geo_cache['obj_pts'][obj_idx])
    plane = pc[obj_idx]
    if not plane:
        return None
    origin, normal = plane[0].copy(), plane[1].copy()
    if normal.dot(view_dir) > 0:
        normal = -normal
    return origin, normal

def pick_edit_face_plane(context, ray_origin, ray_target, bvh_tree):
    obj = context.edit_object
    if not obj or not bvh_tree:
        return None
    matrix = obj.matrix_world
    inv_matrix = matrix.inverted()
    
    local_origin = inv_matrix @ ray_origin
    local_target = (inv_matrix.to_3x3() @ ray_target).normalized()
    
    location, normal, face_idx, dist = bvh_tree.ray_cast(local_origin, local_target)
    if location is not None and face_idx is not None:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        face = bm.faces[face_idx]
        world_origin = matrix @ location
        world_normal = (matrix.to_3x3().inverted().transposed() @ face.normal).normalized()
        if world_normal.dot(ray_target) > 0:
            world_normal = -world_normal
        return world_origin, world_normal
    return None

def V(a):
    return mathutils.Vector((float(a[0]), float(a[1]), float(a[2]) if len(a) > 2 else 0.0))

def nearest_edge_screen(cache, mouse, max_d2):
    a, b = cache['s_e1'], cache['s_e2']
    if len(a) == 0:
        return None
    ok = cache['e1_ok'] & cache['e2_ok']
    ab = b - a
    l2 = np.einsum('ij,ij->i', ab, ab)
    deg = l2 < 1e-9
    t = np.clip(np.einsum('ij,ij->i', mouse - a, ab) / np.where(deg, 1.0, l2), 0.0, 1.0)
    t[deg] = 0.0
    diff = (a + ab * t[:, None]) - mouse
    d2 = np.einsum('ij,ij->i', diff, diff)
    d2[~ok] = np.inf
    i = int(np.argmin(d2))
    return i if d2[i] < max_d2 else None

def closest_on_edge_to_ray(e1, e2, ray_o, ray_d):
    d = e2 - e1
    w0 = e1 - ray_o
    a, b, c = d.dot(d), d.dot(ray_d), ray_d.dot(ray_d)
    dd, e = d.dot(w0), ray_d.dot(w0)
    den = a * c - b * b
    s = 0.0 if (a < 1e-12 or abs(den) < 1e-12) else (b * e - c * dd) / den
    return e1 + d * max(0.0, min(1.0, s))

def perp_candidates(cache, region, rv3d, last_pt):
    key = (cache['screen_key'], tuple(last_pt))
    if cache.get('perp_key') == key:
        return cache['perp_feet'], cache['perp_scr'], cache['perp_ok']
    p = np.array(last_pt, dtype=np.float64)
    e1, e2 = cache['e1'], cache['e2']
    d = e2 - e1
    dd = np.einsum('ij,ij->i', d, d)
    t = np.einsum('ij,ij->i', p - e1, d) / np.where(dd > 1e-12, dd, 1.0)
    foot = e1 + d * t[:, None]
    scr, ok = project_np(region, rv3d, foot)
    far = np.einsum('ij,ij->i', foot - p, foot - p) > 1e-8
    ok = ok & (dd > 1e-12) & (t > 0.001) & (t < 0.999) & far
    cache.update(perp_key=key, perp_feet=foot, perp_scr=scr, perp_ok=ok)
    return foot, scr, ok

def find_snap_target(region, rv3d, mouse_2d, active_points, geo_cache,
                     snap_vertex=True, snap_midpoint=True, snap_edge=False, snap_perp=False,
                     max_dist=14.0, ui_scale=1.0):
    refresh_screen_cache(geo_cache, region, rv3d)
    mouse = np.array((mouse_2d.x, mouse_2d.y))
    max_d2 = max_dist * max_dist
    cands = []

    def add(kind, world, screen, d2):
        if d2 < max_d2:
            cands.append((math.sqrt(d2) + SNAP_BIAS_PX[kind] * ui_scale, kind, world, screen))

    if len(active_points) > 1:
        if snap_vertex:
            for pt in active_points[:-1]:
                s = view3d_utils.location_3d_to_region_2d(region, rv3d, pt)
                if s:
                    add('VERTEX', pt.copy(), s, (s - mouse_2d).length_squared)
        if snap_midpoint:
            for i in range(len(active_points) - 1):
                m = (active_points[i] + active_points[i + 1]) * 0.5
                s = view3d_utils.location_3d_to_region_2d(region, rv3d, m)
                if s:
                    add('MIDPOINT', m, s, (s - mouse_2d).length_squared)

    if snap_vertex:
        hit = nearest_screen(geo_cache['s_verts'], geo_cache['v_ok'], mouse, max_d2)
        if hit:
            i, d2 = hit
            add('VERTEX', V(geo_cache['verts'][i]), V(geo_cache['s_verts'][i]), d2)
    if snap_midpoint:
        hit = nearest_screen(geo_cache['s_mids'], geo_cache['m_ok'], mouse, max_d2)
        if hit:
            i, d2 = hit
            add('MIDPOINT', V(geo_cache['mids'][i]), V(geo_cache['s_mids'][i]), d2)

    if snap_edge:
        i = nearest_edge_screen(geo_cache, mouse, max_d2)
        if i is not None:
            ray_o = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_2d)
            ray_d = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_2d).normalized()
            w = closest_on_edge_to_ray(V(geo_cache['e1'][i]), V(geo_cache['e2'][i]), ray_o, ray_d)
            s = view3d_utils.location_3d_to_region_2d(region, rv3d, w)
            if s:
                add('EDGE', w, s, (s - mouse_2d).length_squared)

    if snap_perp and len(active_points) > 0:
        feet, scr, ok = perp_candidates(geo_cache, region, rv3d, active_points[-1])
        hit = nearest_screen(scr, ok, mouse, max_d2)
        if hit:
            i, d2 = hit
            add('PERPENDICULAR', V(feet[i]), V(scr[i]), d2)

    if not cands:
        return None

    _, kind, world, screen = min(cands, key=lambda c: c[0])
    return {'type': kind, 'world_pos': world, 'screen_pos': screen, 'color': SNAP_COLORS[kind]}

def draw_lines(verts, color, mode='LINES'):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", color)
    batch = batch_for_shader(shader, mode, {"pos": verts})
    batch.draw(shader)

def draw_crosshair(x, y, s, color):
    arm, gap = 14 * s, 4 * s
    draw_lines([
        (x - arm, y), (x - gap, y),
        (x + gap, y), (x + arm, y),
        (x, y - arm), (x, y - gap),
        (x, y + gap), (x, y + arm),
    ], color, 'LINES')

def draw_snap_shape(x, y, s, kind, color):
    r = 7 * s
    if kind == 'VERTEX':
        draw_lines([(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)], color, 'LINE_LOOP')
    elif kind == 'MIDPOINT':
        draw_lines([(x - r, y - r), (x + r, y - r), (x, y + r)], color, 'LINE_LOOP')
    elif kind == 'EDGE':
        draw_lines([(x - r, y + r), (x + r, y + r), (x - r, y - r), (x + r, y - r)], color, 'LINE_LOOP')
    elif kind == 'PERPENDICULAR':
        draw_lines([(x - r, y + r), (x - r, y - r), (x + r, y - r)], color, 'LINE_STRIP')
        draw_lines([(x - r, y), (x, y), (x, y - r)], color, 'LINE_STRIP')
    else:
        draw_lines([(x + math.cos(a * math.pi / 12) * r, y + math.sin(a * math.pi / 12) * r)
                    for a in range(24)], color, 'LINE_LOOP')

def draw_2d_square(x, y, half_size, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("color", color)

    verts = [
        (x - half_size, y - half_size),
        (x + half_size, y - half_size),
        (x + half_size, y + half_size),
        (x - half_size, y + half_size)
    ]
    batch = batch_for_shader(shader, 'TRI_FAN', {"pos": verts})
    batch.draw(shader)


# ==============================================================================
# 3. POLY SKETCH & POLY CUT ENGINES
# ==============================================================================

def draw_callback_3d_sketch(self, context):
    if bpy.context.area != self.area:
        return

    region, rv3d = get_3d_region_and_rv3d(bpy.context)
    if not region:
        return

    gpu.state.blend_set('ALPHA')

    if self.stage == 'PLANE_SELECT':
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        shader.bind()

        grid_positions = []
        grid_colors = []
        
        view_dist = rv3d.view_distance if rv3d else 5.0
        radius = max(0.5, view_dist * 0.3)
        step = radius / 6.0
        steps_count = int(radius / step)
        
        for i in range(-steps_count, steps_count + 1):
            offset = i * step
            a = max(0.0, 1.0 - (abs(offset) / radius) ** 1.5) * 0.65
            if a < 0.05:
                continue

            u_p1 = self.plane_origin + self.plane_u * offset - self.plane_v * radius
            u_p2 = self.plane_origin + self.plane_u * offset + self.plane_v * radius
            v_p1 = self.plane_origin - self.plane_u * radius + self.plane_v * offset
            v_p2 = self.plane_origin + self.plane_u * radius + self.plane_v * offset

            grid_positions.extend([u_p1, u_p2, v_p1, v_p2])
            grid_colors.extend([(1.0, 1.0, 1.0, a), (1.0, 1.0, 1.0, a), (1.0, 1.0, 1.0, a), (1.0, 1.0, 1.0, a)])

        if grid_positions:
            gpu.state.line_width_set(1.5)
            batch_grid = batch_for_shader(shader, 'LINES', {"pos": grid_positions, "color": grid_colors})
            batch_grid.draw(shader)

    elif self.stage == 'DRAWING' and len(self.points) > 0:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        shader.bind()
        shader.uniform_float("viewportSize", (float(region.width), float(region.height)))

        line_color = (1.0, 0.35, 0.1, 1.0) if self.tool_mode == 'CUT' else (0.1, 0.8, 1.0, 1.0)
        shader.uniform_float("lineWidth", 3.5)
        shader.uniform_float("color", line_color)
        batch_lines = batch_for_shader(shader, 'LINE_STRIP', {"pos": self.points})
        batch_lines.draw(shader)

        if self.active_pt:
            color = (1.0, 0.25, 0.25, 1.0) if self.axis_snapped else (0.2, 0.95, 0.35, 1.0)
            shader.uniform_float("lineWidth", 3.0)
            shader.uniform_float("color", color)
            batch_guide = batch_for_shader(shader, 'LINE_STRIP', {"pos": [self.points[-1], self.active_pt]})
            batch_guide.draw(shader)

        shader_pt = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader_pt.bind()
        gpu.state.point_size_set(8.0)
        shader_pt.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_pts = batch_for_shader(shader_pt, 'POINTS', {"pos": self.points})
        batch_pts.draw(shader_pt)

        gpu.state.point_size_set(14.0)
        shader_pt.uniform_float("color", (1.0, 0.5, 0.0, 1.0))
        batch_start = batch_for_shader(shader_pt, 'POINTS', {"pos": [self.points[0]]})
        batch_start.draw(shader_pt)

    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)

def draw_callback_2d_sketch(self, context):
    if bpy.context.area != self.area:
        return

    region, rv3d = get_3d_region_and_rv3d(bpy.context)
    if not region or not rv3d:
        return

    ui_scale = get_ui_scale(bpy.context)
    target_2d = self.mouse_2d.copy()

    if self.active_pt:
        s_pos = view3d_utils.location_3d_to_region_2d(region, rv3d, self.active_pt)
        if s_pos:
            target_2d = s_pos

    gpu.state.blend_set('ALPHA')
    cx, cy = target_2d.x, target_2d.y

    if self.snap_info:
        draw_snap_shape(cx, cy, ui_scale, self.snap_info['type'], self.snap_info['color'])
        draw_crosshair(cx, cy, ui_scale, (1.0, 1.0, 1.0, 0.6))
    else:
        col = (1.0, 0.25, 0.25, 0.95) if self.axis_snapped else (1.0, 1.0, 1.0, 0.95)
        draw_crosshair(cx, cy, ui_scale, col)
        draw_2d_square(cx, cy, 1.5 * ui_scale, col)

    font_id = 0
    blf.size(font_id, int(13 * ui_scale))

    if self.stage == 'DRAWING' and len(self.points) > 0 and self.active_pt:
        dist = (self.active_pt - self.points[-1]).length
        dist_str = format_unit_length(bpy.context, dist)
        
        blf.position(font_id, int(cx + 20 * ui_scale), int(cy + 12 * ui_scale), 0)
        if self.typed_buffer:
            blf.color(font_id, 1.0, 0.5, 0.0, 1.0)
            blf.draw(font_id, f"Length: {self.typed_buffer} [Press ENTER]")
        else:
            blf.color(font_id, 0.95, 0.95, 0.95, 1.0)
            snap_lbl = f" ({self.snap_info['type']})" if self.snap_info else (" (Axis Lock)" if self.axis_snapped else "")
            blf.draw(font_id, f"{dist_str}{snap_lbl}")

        blf.position(font_id, 20, 40, 0)
        blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
        tool_label = "Poly Cut" if self.tool_mode == 'CUT' else "Poly Sketch"
        blf.draw(font_id, f"{tool_label}: Click: Add Point | Ctrl+Z: Undo Point | Type Unit + Enter: Length | Enter: Finish")

    if getattr(self, "snap_note", ""):
        blf.position(font_id, 20, 62, 0)
        blf.color(font_id, 1.0, 0.6, 0.2, 1.0)
        blf.draw(font_id, self.snap_note)

    gpu.state.blend_set('NONE')

def compute_sketch_segments(points, closed, plane_origin, plane_u, plane_v):
    n = len(points)
    if n < 2:
        return []
    segments = []
    for i in range(n - 1):
        segments.append((points[i].copy(), points[i+1].copy()))
    if closed and n >= 3:
        segments.append((points[-1].copy(), points[0].copy()))
    return segments

def split_and_build_sketch_bmesh(points, closed, plane_origin, plane_normal, plane_u, plane_v):
    raw_segments = compute_sketch_segments(points, closed, plane_origin, plane_u, plane_v)

    def to_2d(p3d):
        v = p3d - plane_origin
        return mathutils.Vector((v.dot(plane_u), v.dot(plane_v)))

    def to_3d(p2d):
        return plane_origin + plane_u * p2d.x + plane_v * p2d.y

    segment_splits = [[] for _ in range(len(raw_segments))]

    for i in range(len(raw_segments)):
        p1_3d, p2_3d = raw_segments[i]
        p1_2d, p2_2d = to_2d(p1_3d), to_2d(p2_3d)
        dir_2d = p2_2d - p1_2d
        len_2d_sq = dir_2d.length_squared

        segment_splits[i].append((0.0, p1_3d))
        segment_splits[i].append((1.0, p2_3d))

        if len_2d_sq < 1e-8:
            continue

        for j in range(i + 1, len(raw_segments)):
            q1_3d, q2_3d = raw_segments[j]
            q1_2d, q2_2d = to_2d(q1_3d), to_2d(q2_3d)
            
            hit_2d = mathutils.geometry.intersect_line_line_2d(p1_2d, p2_2d, q1_2d, q2_2d)
            if hit_2d:
                t_i = (hit_2d - p1_2d).dot(dir_2d) / len_2d_sq
                dir_j = q2_2d - q1_2d
                len_j_sq = dir_j.length_squared
                if len_j_sq < 1e-8:
                    continue
                t_j = (hit_2d - q1_2d).dot(dir_j) / len_j_sq

                eps = 1e-4
                if -eps <= t_i <= 1.0 + eps and -eps <= t_j <= 1.0 + eps:
                    t_i_clamped = max(0.0, min(1.0, t_i))
                    t_j_clamped = max(0.0, min(1.0, t_j))
                    hit_3d = to_3d(hit_2d)

                    segment_splits[i].append((t_i_clamped, hit_3d))
                    segment_splits[j].append((t_j_clamped, hit_3d))

    bm = bmesh.new()
    vert_map = {}

    def get_or_create_vert(p3d):
        key = (round(p3d.x, 4), round(p3d.y, 4), round(p3d.z, 4))
        if key in vert_map and vert_map[key].is_valid:
            return vert_map[key]
        v = bm.verts.new(p3d)
        vert_map[key] = v
        return v

    for i, splits in enumerate(segment_splits):
        splits.sort(key=lambda item: item[0])

        for k in range(len(splits) - 1):
            t1, pt1 = splits[k]
            t2, pt2 = splits[k + 1]

            if (t2 - t1) > 1e-4:
                v1 = get_or_create_vert(pt1)
                v2 = get_or_create_vert(pt2)
                if v1 != v2:
                    if not bm.edges.get((v1, v2)):
                        bm.edges.new((v1, v2))

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.001)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    try:
        bmesh.ops.edgenet_fill(bm, edges=bm.edges[:])
    except Exception as ex:
        print(f"[PolySketch Notice] edgenet_fill: {ex}")

    return bm

def merge_sketch_into_edit_mesh(edit_obj, segments, plane_origin, plane_normal, plane_u, plane_v):
    if not segments or not edit_obj:
        return

    bm = bmesh.from_edit_mesh(edit_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for v in bm.verts: v.select = False
    for e in bm.edges: e.select = False
    for f in bm.faces: f.select = False

    mw = edit_obj.matrix_world
    imw = mw.inverted()

    unit_scale = bpy.context.scene.unit_settings.scale_length
    obj_scale = max(mw.to_scale()) if max(mw.to_scale()) > 1e-6 else 1.0
    local_tol = (1e-3 * unit_scale) / obj_scale

    local_origin = imw @ plane_origin
    local_normal = (imw.to_3x3() @ plane_normal).normalized()
    local_u = (imw.to_3x3() @ plane_u).normalized()
    local_v = (imw.to_3x3() @ plane_v).normalized()

    def to_2d(p3d):
        v = p3d - local_origin
        return mathutils.Vector((v.dot(local_u), v.dot(local_v)))

    def to_3d(p2d):
        return local_origin + local_u * p2d.x + local_v * p2d.y

    coplanar_faces = []
    for f in bm.faces:
        if abs(f.normal.dot(local_normal)) > 0.999:
            if all(abs((v.co - local_origin).dot(local_normal)) < local_tol * 10.0 for v in f.verts):
                coplanar_faces.append(f)

    if not coplanar_faces:
        return

    raw_segs_2d = []
    for p1_w, p2_w in segments:
        p1_l, p2_l = imw @ p1_w, imw @ p2_w
        raw_segs_2d.append((p1_l, p2_l, to_2d(p1_l), to_2d(p2_l)))

    num_segs = len(raw_segs_2d)
    seg_splits = [[] for _ in range(num_segs)]

    for i in range(num_segs):
        p1_l, p2_l, p1_2d, p2_2d = raw_segs_2d[i]
        dir_2d = p2_2d - p1_2d
        len_sq = dir_2d.length_squared
        seg_splits[i].append((0.0, p1_l, p1_2d))
        seg_splits[i].append((1.0, p2_l, p2_2d))

        if len_sq < 1e-8:
            continue

        for j in range(i + 1, num_segs):
            q1_l, q2_l, q1_2d, q2_2d = raw_segs_2d[j]
            hit_2d = mathutils.geometry.intersect_line_line_2d(p1_2d, p2_2d, q1_2d, q2_2d)
            if hit_2d:
                t_i = (hit_2d - p1_2d).dot(dir_2d) / len_sq
                dir_j = q2_2d - q1_2d
                len_j_sq = dir_j.length_squared
                if len_j_sq > 1e-8:
                    t_j = (hit_2d - q1_2d).dot(dir_j) / len_j_sq
                    if -1e-4 <= t_i <= 1.0001 and -1e-4 <= t_j <= 1.0001:
                        t_i_c = max(0.0, min(1.0, t_i))
                        t_j_c = max(0.0, min(1.0, t_j))
                        hit_3d = to_3d(hit_2d)
                        seg_splits[i].append((t_i_c, hit_3d, hit_2d))
                        seg_splits[j].append((t_j_c, hit_3d, hit_2d))

    subdivided_segments = []
    for i, splits in enumerate(seg_splits):
        splits.sort(key=lambda item: item[0])
        for k in range(len(splits) - 1):
            t1, pt1_3d, pt1_2d = splits[k]
            t2, pt2_3d, pt2_2d = splits[k + 1]
            if (t2 - t1) > 1e-4:
                subdivided_segments.append((pt1_3d, pt2_3d, pt1_2d, pt2_2d))

    coplanar_edges = list({e for f in coplanar_faces for e in f.edges})
    created_verts = []

    def get_or_split_vert(p3d, p2d, tol=local_tol):
        for v in created_verts:
            if v.is_valid and (v.co - p3d).length < tol:
                return v
        for f in coplanar_faces:
            for v in f.verts:
                if (v.co - p3d).length < tol:
                    return v
        for e in list(coplanar_edges):
            if not e.is_valid:
                continue
            e_p1_2d, e_p2_2d = to_2d(e.verts[0].co), to_2d(e.verts[1].co)
            hit_2d = mathutils.geometry.intersect_point_line(p2d, e_p1_2d, e_p2_2d)
            if hit_2d:
                closest_pt_2d, factor = hit_2d
                if (closest_pt_2d - p2d).length < tol and 0.001 < factor < 0.999:
                    e_new, v_split = bmesh.utils.edge_split(e, e.verts[0], factor)
                    coplanar_edges.append(e_new)
                    created_verts.append(v_split)
                    return v_split
        new_v = bm.verts.new(p3d)
        created_verts.append(new_v)
        return new_v

    new_edges = []
    for p1_l, p2_l, p1_2d, p2_2d in subdivided_segments:
        pts_along = [(0.0, p1_l, p1_2d), (1.0, p2_l, p2_2d)]
        seg_dir_2d = p2_2d - p1_2d
        seg_len_2d_sq = seg_dir_2d.length_squared
        if seg_len_2d_sq < 1e-8:
            continue

        for e in list(coplanar_edges):
            if not e.is_valid:
                continue
            e_p1_2d, e_p2_2d = to_2d(e.verts[0].co), to_2d(e.verts[1].co)
            hit_2d = mathutils.geometry.intersect_line_line_2d(p1_2d, p2_2d, e_p1_2d, e_p2_2d)
            if hit_2d:
                t_seg = (hit_2d - p1_2d).dot(seg_dir_2d) / seg_len_2d_sq
                e_dir_2d = e_p2_2d - e_p1_2d
                e_len_2d_sq = e_dir_2d.length_squared
                if e_len_2d_sq > 1e-8:
                    t_edge = (hit_2d - e_p1_2d).dot(e_dir_2d) / e_len_2d_sq
                    if -1e-4 <= t_seg <= 1.0001 and -1e-4 <= t_edge <= 1.0001:
                        t_seg_clamped = max(0.0, min(1.0, t_seg))
                        hit_3d = to_3d(hit_2d)
                        pts_along.append((t_seg_clamped, hit_3d, hit_2d))

        pts_along.sort(key=lambda item: item[0])

        for k in range(len(pts_along) - 1):
            t1, pt1_3d, pt1_2d = pts_along[k]
            t2, pt2_3d, pt2_2d = pts_along[k + 1]
            if (t2 - t1) > 1e-4:
                v1 = get_or_split_vert(pt1_3d, pt1_2d)
                v2 = get_or_split_vert(pt2_3d, pt2_2d)
                if v1 != v2:
                    existing_e = bm.edges.get((v1, v2))
                    e_curr = existing_e if existing_e else bm.edges.new((v1, v2))
                    if e_curr not in new_edges:
                        new_edges.append(e_curr)

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    for f in list(coplanar_faces):
        if not f.is_valid:
            continue
        f_poly_2d = [to_2d(v.co) for v in f.verts]
        face_cut_edges = []
        for e in new_edges:
            if not e.is_valid or e in f.edges:
                continue
            mid_3d = (e.verts[0].co + e.verts[1].co) * 0.5
            mid_2d = to_2d(mid_3d)
            if is_point_in_polygon_2d(mid_2d, f_poly_2d):
                face_cut_edges.append(e)

        if face_cut_edges:
            try:
                bmesh.utils.face_split_edgenet(f, face_cut_edges)
            except Exception as ex:
                print(f"[PolyCut Notice] face_split_edgenet: {ex}")

    for e in new_edges:
        if e.is_valid:
            e.select = True
            e.verts[0].select = True
            e.verts[1].select = True

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    
    affected_verts = [v for v in created_verts if v.is_valid]
    if affected_verts:
        bmesh.ops.remove_doubles(bm, verts=affected_verts, dist=local_tol)

    bmesh.update_edit_mesh(edit_obj.data)

class CAD_OT_2d_base(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}
    tool_mode = 'SKETCH'

    def finish(self, context, closed):
        try:
            self.finalize_tool(context, closed=closed)
            result = {'FINISHED'}
        except Exception as ex:
            self.report({'ERROR'}, f"{self.bl_label} failed: {ex}")
            result = {'CANCELLED'}
        self.cleanup(context)
        return result

    def modal(self, context, event):
        tag_redraw(context)

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'}:
            return {'PASS_THROUGH'}
        if event.alt or (event.type == 'MOUSEMOVE' and event.oskey):
            return {'PASS_THROUGH'}

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if getattr(self, 'typed_buffer', ''):
                self.typed_buffer = ""
                return {'RUNNING_MODAL'}
            self.cleanup(context)
            return {'CANCELLED'}

        region, rv3d = get_3d_region_and_rv3d(context)
        if not region or not rv3d:
            return {'PASS_THROUGH'}

        ui_scale = get_ui_scale(context)
        snap_pixel_dist = SNAP_RADIUS_PX * ui_scale
        close_dist = CLOSE_RADIUS_PX * ui_scale

        self.mouse_2d = mathutils.Vector((event.mouse_region_x, event.mouse_region_y))

        tool_settings = context.tool_settings
        is_snap_enabled = tool_settings.use_snap != event.ctrl
        snap_elements = set(tool_settings.snap_elements)

        snap_vertex_active   = 'VERTEX' in snap_elements
        snap_midpoint_active = 'EDGE_MIDPOINT' in snap_elements or 'MIDPOINT' in snap_elements
        snap_edge_active     = 'EDGE' in snap_elements or 'EDGE_PERPENDICULAR' in snap_elements
        snap_perp_active     = 'EDGE_PERPENDICULAR' in snap_elements or 'PERPENDICULAR' in snap_elements

        supported = (snap_vertex_active or snap_midpoint_active or snap_edge_active or snap_perp_active)
        self.snap_note = ""
        if is_snap_enabled and not supported:
            self.snap_note = "Snap mode not supported: use Vertex, Edge, Midpoint or Perpendicular"

        if self.stage == 'PLANE_SELECT':
            depsgraph = context.evaluated_depsgraph_get()
            ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse_2d)
            ray_target = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse_2d)
            
            edit_plane = None
            if self.tool_mode == 'CUT':
                edit_plane = pick_edit_face_plane(context, ray_origin, ray_target, self.edit_bvh)
                self.hover_ok = edit_plane is not None
                if edit_plane:
                    self.plane_origin, self.plane_normal = edit_plane
                else:
                    self.plane_origin = ray_origin + ray_target * 5.0
                    self.plane_normal = mathutils.Vector((0, 0, 1))
            else:
                result, location, normal, idx, obj, matrix = context.scene.ray_cast(depsgraph, ray_origin, ray_target)
                if result:
                    self.plane_origin = location
                    self.plane_normal = normal.normalized()
                    if self.plane_normal.dot(ray_target) > 0:
                        self.plane_normal = -self.plane_normal
                else:
                    wire = pick_wire_plane(region, rv3d, self.mouse_2d, ray_target.normalized(), self.geo_cache)
                    if wire:
                        self.plane_origin, self.plane_normal = wire
                    else:
                        denom = ray_target.z
                        if abs(denom) > 1e-6:
                            t = -ray_origin.z / denom
                            if t > 0:
                                hit_pt = ray_origin + t * ray_target
                                self.plane_origin = mathutils.Vector((hit_pt.x, hit_pt.y, 0.0))
                            else:
                                self.plane_origin = mathutils.Vector((0.0, 0.0, 0.0))
                        else:
                            self.plane_origin = mathutils.Vector((0.0, 0.0, 0.0))
                        self.plane_normal = mathutils.Vector((0.0, 0.0, 1.0))

            self.plane_u, self.plane_v = compute_plane_basis(self.plane_normal)

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                if self.tool_mode == 'CUT' and not getattr(self, 'hover_ok', False):
                    return {'RUNNING_MODAL'}
                self.stage = 'DRAWING'
                return {'RUNNING_MODAL'}
            elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                if self.tool_mode == 'CUT':
                    if not getattr(self, 'hover_ok', False):
                        return {'RUNNING_MODAL'}
                    self.stage = 'DRAWING'
                    return {'RUNNING_MODAL'}
                else:
                    self.plane_origin = mathutils.Vector((0, 0, 0))
                    self.plane_normal = mathutils.Vector((0, 0, 1))
                    self.plane_u, self.plane_v = compute_plane_basis(self.plane_normal)
                    self.stage = 'DRAWING'
                    return {'RUNNING_MODAL'}
            
            return {'RUNNING_MODAL'}

        elif self.stage == 'DRAWING':
            plane_hit = raycast_plane_intersection(region, rv3d, self.mouse_2d, self.plane_origin, self.plane_normal)
            self.axis_snapped = False
            self.is_closing = False
            self.snap_info = None

            if len(self.points) >= 3:
                v0_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, self.points[0])
                if v0_2d and (v0_2d - self.mouse_2d).length < close_dist:
                    plane_hit = self.points[0].copy()
                    self.is_closing = True
                    self.snap_info = {'type': 'CLOSE_LOOP', 'color': (1.0, 0.5, 0.0, 1.0)}

            if not self.is_closing and is_snap_enabled and supported:
                snap_hit = find_snap_target(
                    region, rv3d, self.mouse_2d, self.points, self.geo_cache,
                    snap_vertex=snap_vertex_active,
                    snap_midpoint=snap_midpoint_active,
                    snap_edge=snap_edge_active,
                    snap_perp=snap_perp_active,
                    max_dist=snap_pixel_dist,
                    ui_scale=ui_scale
                )
                if snap_hit:
                    w_pos = snap_hit['world_pos']
                    vec = w_pos - self.plane_origin
                    dist_along_norm = vec.dot(self.plane_normal)
                    plane_hit = w_pos - self.plane_normal * dist_along_norm
                    
                    proj_2d = view3d_utils.location_3d_to_region_2d(region, rv3d, plane_hit)
                    if proj_2d and (proj_2d - self.mouse_2d).length <= snap_pixel_dist * 2.5:
                        self.snap_info = snap_hit

            if not self.snap_info and len(self.points) > 0:
                last_pt = self.points[-1]
                vec = plane_hit - last_pt
                
                u_dist = vec.dot(self.plane_u)
                v_dist = vec.dot(self.plane_v)
                
                snap_thresh = 0.07 * vec.length
                if abs(u_dist) < snap_thresh:
                    plane_hit = last_pt + self.plane_v * v_dist
                    self.axis_snapped = True
                elif abs(v_dist) < snap_thresh:
                    plane_hit = last_pt + self.plane_u * u_dist
                    self.axis_snapped = True

            self.typed_ok = False
            if len(self.points) > 0 and self.typed_buffer:
                parsed_val = parse_unit_expression(context, self.typed_buffer)
                if parsed_val is not None and parsed_val > 0:
                    dir_vec = plane_hit - self.points[-1]
                    if dir_vec.length_squared > 1e-6:
                        dir_vec.normalize()
                    else:
                        dir_vec = self.plane_u.copy()
                    plane_hit = self.points[-1] + dir_vec * parsed_val
                    self.typed_ok = True
                    self.is_closing = False
                    self.snap_info = None

            self.active_pt = plane_hit

            if event.value == 'PRESS':
                if event.type == 'Z' and (event.ctrl or event.oskey):
                    if self.typed_buffer:
                        self.typed_buffer = ""
                    elif len(self.points) > 0:
                        self.points.pop()
                    return {'RUNNING_MODAL'}
                elif event.type == 'BACK_SPACE':
                    self.typed_buffer = self.typed_buffer[:-1]
                    return {'RUNNING_MODAL'}
                elif event.unicode and event.unicode in "0123456789.+-*/'\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ":
                    self.typed_buffer += event.unicode
                    return {'RUNNING_MODAL'}

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                if self.typed_buffer:
                    if self.typed_ok:
                        self.points.append(self.active_pt.copy())
                    self.typed_buffer = ""
                    return {'RUNNING_MODAL'}
                if self.is_closing:
                    return self.finish(context, closed=True)
                self.points.append(self.active_pt.copy())
                return {'RUNNING_MODAL'}

            elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                if self.typed_buffer:
                    if self.typed_ok:
                        self.points.append(self.active_pt.copy())
                    self.typed_buffer = ""
                    return {'RUNNING_MODAL'}
                if len(self.points) >= 2:
                    return self.finish(context, closed=False)

            return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        self.area = context.area
        self.region = context.region
        self.rv3d = context.space_data.region_3d if context.space_data else None

        self.stage = 'PLANE_SELECT'
        self.points = []
        self.active_pt = None
        self.typed_buffer = ""
        self.axis_snapped = False
        self.is_closing = False
        self.snap_info = None
        self.snap_note = ""
        self.typed_ok = False
        self.hover_ok = False
        self.mouse_2d = mathutils.Vector((event.mouse_region_x, event.mouse_region_y))
        
        self.plane_origin = mathutils.Vector((0, 0, 0))
        self.plane_normal = mathutils.Vector((0, 0, 1))
        self.plane_u, self.plane_v = compute_plane_basis(self.plane_normal)

        self.edit_bvh = None

        if self.tool_mode == 'CUT' and context.edit_object and context.edit_object.type == 'MESH':
            bm_temp = bmesh.from_edit_mesh(context.edit_object.data)
            bm_temp.faces.ensure_lookup_table()
            self.edit_bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm_temp)

        self.geo_cache = build_geo_cache(context)

        context.window.cursor_modal_set('NONE')

        args = (self, context)
        self._handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d_sketch, args, 'WINDOW', 'POST_VIEW')
        self._handle_2d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_2d_sketch, args, 'WINDOW', 'POST_PIXEL')
        
        global _active_draw_handlers
        _active_draw_handlers.append((self._handle_3d, self._handle_2d))

        context.window_manager.modal_handler_add(self)
        tag_redraw(context)
        return {'RUNNING_MODAL'}

    def cleanup(self, context):
        context.window.cursor_modal_restore()

        global _active_draw_handlers
        pair = (getattr(self, "_handle_3d", None), getattr(self, "_handle_2d", None))
        if pair in _active_draw_handlers:
            _active_draw_handlers.remove(pair)

        if getattr(self, "_handle_3d", None) is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_3d, 'WINDOW')
            except Exception:
                pass
        if getattr(self, "_handle_2d", None) is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle_2d, 'WINDOW')
            except Exception:
                pass
        tag_redraw(context)

class MESH_OT_poly_sketch_2d(CAD_OT_2d_base):
    """Interactive CAD 2D Sketch tool that creates planar geometry objects."""
    bl_idname = "mesh.poly_sketch_2d"
    bl_label = "Poly Sketch"
    tool_mode = 'SKETCH'

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def finalize_tool(self, context, closed=False):
        if len(self.points) < 2:
            return

        bm = split_and_build_sketch_bmesh(
            self.points, 
            closed, 
            self.plane_origin, 
            self.plane_normal, 
            self.plane_u, 
            self.plane_v
        )

        mesh_data = bpy.data.meshes.new("2D_PolySketch")
        bm.to_mesh(mesh_data)
        bm.free()

        obj = bpy.data.objects.new("2D_Sketch_Object", mesh_data)
        context.collection.objects.link(obj)
        
        bbox_center = 0.125 * sum((mathutils.Vector(b) for b in obj.bound_box), mathutils.Vector())
        obj.location = bbox_center
        mesh_data.transform(mathutils.Matrix.Translation(-bbox_center))

        for o in context.selected_objects: o.select_set(False)
        context.view_layer.objects.active = obj
        obj.select_set(True)

class MESH_OT_poly_cut_2d(CAD_OT_2d_base):
    """Interactive CAD 2D Cut tool that splits coplanar mesh faces in Edit Mode."""
    bl_idname = "mesh.poly_cut_2d"
    bl_label = "Poly Cut"
    tool_mode = 'CUT'

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and context.edit_object is not None and context.edit_object.type == 'MESH'

    def finalize_tool(self, context, closed=False):
        if len(self.points) < 2:
            return

        segments = compute_sketch_segments(self.points, closed, self.plane_origin, self.plane_u, self.plane_v)
        merge_sketch_into_edit_mesh(
            context.edit_object, 
            segments, 
            self.plane_origin, 
            self.plane_normal, 
            self.plane_u, 
            self.plane_v
        )


# ==============================================================================
# 4. POLY OFFSET ENGINE (v1.2.0)
# ==============================================================================

_line_isect = lambda p, r, q, s: (p + r * (((q - p).x * s.y - (q - p).y * s.x) / (r.x * s.y - r.y * s.x))) if abs(r.x * s.y - r.y * s.x) >= 1e-12 else None
MITER_LIMIT = 50.0

def offset_polyline_2d(pts, dist, closed):
    n = len(pts)
    if n < 2:
        return list(pts)
    if closed and n >= 3:
        a2 = sum(pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y for i in range(n))
        w = 1.0 if a2 >= 0 else -1.0
    else:
        w = 1.0

    num = n if closed else n - 1
    segs = []
    for i in range(num):
        p1, p2 = pts[i], pts[(i + 1) % n]
        d = (p2 - p1).normalized() if (p2 - p1).length_squared > 1e-12 else mathutils.Vector((1, 0))
        segs.append((p1 + mathutils.Vector((d.y, -d.x)) * (w * dist), d))

    out = []
    for i in range(n):
        if not closed and i == 0:
            out.append(segs[0][0])
            continue
        if not closed and i == n - 1:
            a, d = segs[-1]
            out.append(a + d * (pts[-1] - pts[-2]).length)
            continue

        a0, d0 = segs[(i - 1) % num]
        a1, d1 = segs[i % num]
        
        den = d0.x * d1.y - d0.y * d1.x
        if abs(den) < 1e-4:
            if d0.dot(d1) > 0.0:
                hit = pts[i] + (a0 - pts[(i - 1) % n])  # Straight run collinear stabilization
            else:
                hit = a1                                # Fold-back corner
        else:
            hit = _line_isect(a0, d0, a1, d1)

        if hit is None:
            hit = a1

        off = hit - pts[i]
        lim = MITER_LIMIT * abs(dist)
        if lim > 0 and off.length > lim:
            hit = pts[i] + off.normalized() * lim
        out.append(hit)

    return out

def _nearest_on_polyline_2d(pt, pts, closed):
    n = len(pts)
    best_d2, best_sign = None, 1.0
    count = n if closed else n - 1
    for i in range(count):
        a, b = pts[i], pts[(i + 1) % n]
        ab = b - a
        l2 = ab.length_squared
        if l2 < 1e-12:
            continue
        t = max(0.0, min(1.0, (pt - a).dot(ab) / l2))
        proj = a + ab * t
        d = pt - proj
        d2 = d.length_squared
        sign = 1.0 if (ab.x * d.y - ab.y * d.x) >= 0.0 else -1.0
        if best_d2 is None or d2 < best_d2:
            best_d2, best_sign = d2, sign
    if best_d2 is None:
        return (pt - pts[0]).length, 1.0
    return math.sqrt(best_d2), best_sign

def signed_mouse_distance_2d(mouse, pts, closed):
    dist, side = _nearest_on_polyline_2d(mouse, pts, closed)
    if closed:
        return -dist if is_point_in_polygon_2d(mouse, pts) else dist
    return -side * dist

def calc_max_inward_clamp(pts_2d):
    n = len(pts_2d)
    if n < 3:
        return float('inf')

    a2 = sum(pts_2d[i].x * pts_2d[(i + 1) % n].y - pts_2d[(i + 1) % n].x * pts_2d[i].y for i in range(n))
    w = 1.0 if a2 >= 0 else -1.0

    def corner_miter(p_a, p_b, p_c):
        d1 = (p_b - p_a).normalized() if (p_b - p_a).length_squared > 1e-12 else mathutils.Vector((1, 0))
        d2 = (p_c - p_b).normalized() if (p_c - p_b).length_squared > 1e-12 else mathutils.Vector((1, 0))
        n1 = mathutils.Vector((d1.y, -d1.x)) * w
        n2 = mathutils.Vector((d2.y, -d2.x)) * w
        b = n1 + n2
        b = b.normalized() if b.length_squared > 1e-6 else n2
        den = b.dot(n2)
        m = 1.0 / den if abs(den) > 0.1 else 1.0
        return b * max(-3.0, min(3.0, m))

    v_corners = []
    for i in range(n):
        p_prev, p_curr, p_next = pts_2d[(i - 1) % n], pts_2d[i], pts_2d[(i + 1) % n]
        v_corners.append(corner_miter(p_prev, p_curr, p_next))

    max_d = float('inf')
    for i in range(n):
        p_curr, p_next = pts_2d[i], pts_2d[(i + 1) % n]
        seg = p_next - p_curr
        L = seg.length
        if L < 1e-6:
            continue
        d_edge = seg / L
        v_curr = v_corners[i]
        v_next = v_corners[(i + 1) % n]

        rate = (v_curr - v_next).dot(d_edge)
        if rate < -1e-6:
            t_c = L / -rate
            if 0 < t_c < max_d:
                max_d = t_c

    return max_d

def chain_adjacent_faces(bm, vert_indices, closed):
    n = len(vert_indices)
    out = []
    seen = set()
    for i in range(n if closed else n - 1):
        v1 = bm.verts[vert_indices[i]]
        v2 = bm.verts[vert_indices[(i + 1) % n]]
        e = bm.edges.get((v1, v2))
        if e and is_valid_elem(e):
            for f in e.link_faces:
                if is_valid_elem(f) and f.index not in seen:
                    seen.add(f.index)
                    out.append(f)
    return out

def pick_face_under_cursor(cands, chain_pts, ray_o, ray_d):
    if not cands or len(chain_pts) < 2:
        return None

    extent = max((p - chain_pts[0]).length for p in chain_pts)
    plane_tol = 1e-4 * extent + 1e-6

    best, best_t = None, None
    num_pts = len(chain_pts)

    for n, c in cands:
        # Filter candidate planes that do not contain the whole chain
        if any(abs((p - c).dot(n)) > plane_tol for p in chain_pts):
            continue

        denom = ray_d.dot(n)
        if abs(denom) < 1e-6:
            continue
        t = (c - ray_o).dot(n) / denom
        if t <= 0:
            continue
        hit = ray_o + ray_d * t

        best_seg_idx = 0
        best_seg_d2 = float('inf')
        for i in range(num_pts - 1):
            p1, p2 = chain_pts[i], chain_pts[i + 1]
            seg = p2 - p1
            l2 = seg.length_squared
            if l2 < 1e-12:
                continue
            fac = max(0.0, min(1.0, (hit - p1).dot(seg) / l2))
            proj = p1 + seg * fac
            d2 = (hit - proj).length_squared
            if d2 < best_seg_d2:
                best_seg_d2 = d2
                best_seg_idx = i

        p1, p2 = chain_pts[best_seg_idx], chain_pts[best_seg_idx + 1]
        edge_dir = (p2 - p1).normalized() if (p2 - p1).length_squared > 1e-12 else mathutils.Vector((1, 0, 0))

        inward = c - p1
        inward -= edge_dir * inward.dot(edge_dir)
        inward -= n * inward.dot(n)
        if inward.length < 1e-9:
            continue
        inward = inward.normalized()

        if (hit - p1).dot(inward) <= 0:
            continue

        if best is None or t < best_t:
            best = (c, n)
            best_t = t

    return best

def collect_offset_chains(bm, mw):
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()

    sel_edges = [e for e in bm.edges if is_valid_elem(e) and e.select]
    sel_faces = [f for f in bm.faces if is_valid_elem(f) and f.select]

    if sel_faces:
        boundary = [e for e in sel_edges if sum(1 for f in e.link_faces if f.select) == 1]
        if boundary:
            sel_edges = boundary

    if not sel_edges:
        return []

    adj = {}
    for e in sel_edges:
        v1, v2 = e.verts
        adj.setdefault(v1, []).append((v2, e))
        adj.setdefault(v2, []).append((v1, e))

    remaining = set(sel_edges)
    chains = []
    safety = len(sel_edges) + 4
    while remaining:
        start = None
        for v, nbrs in adj.items():
            if sum(1 for (_, e) in nbrs if e in remaining) == 1:
                start = v
                break
        if start is None:
            start = next(iter(remaining)).verts[0]

        chain_verts = [start]
        closed = False
        cur = start
        while len(chain_verts) < safety:
            nxt = None
            for (nv, e) in adj.get(cur, []):
                if e in remaining:
                    nxt = (nv, e)
                    break
            if nxt is None:
                break
            nv, e = nxt
            remaining.discard(e)
            if nv == start:
                closed = True
                break
            chain_verts.append(nv)
            cur = nv

        if len(chain_verts) >= 2:
            chains.append({
                'vert_indices': [v.index for v in chain_verts],
                'pts_3d': [mw @ v.co.copy() for v in chain_verts],
                'closed': closed
            })

    return chains

def loop_edges(bm, verts, closed):
    n = len(verts)
    out = []
    num = n if closed else n - 1
    for i in range(num):
        e = bm.edges.get((verts[i], verts[(i + 1) % n]))
        if e and is_valid_elem(e):
            out.append(e)
    return out

def face_is_isolated(face):
    return all(len(e.link_faces) == 1 for e in face.edges if is_valid_elem(e))

def replace_face_with_ring(bm, face, orig_verts, off_verts):
    ref_n = face.normal.copy()
    bmesh.ops.delete(bm, geom=[face], context='FACES_ONLY')

    def make(vs):
        if len(set(vs)) < 3:
            return None
        try:
            f = bm.faces.new(vs)
        except ValueError as ex:
            print(f"[PolyOffset Notice] Face creation warning: {ex}")
            return None
        f.normal_update()
        if f.normal.dot(ref_n) < 0:
            f.normal_flip()
        return f

    n = len(orig_verts)
    inner = make(list(off_verts))
    ring = [make([orig_verts[i], orig_verts[(i + 1) % n],
                  off_verts[(i + 1) % n], off_verts[i]]) for i in range(n)]

    return inner, [q for q in ring if q and is_valid_elem(q)]

def add_ring_only(bm, face, orig_verts, off_verts):
    ref_n = face.normal.copy()
    n = len(orig_verts)
    ring = []
    for i in range(n):
        vs = [orig_verts[i], orig_verts[(i + 1) % n], off_verts[(i + 1) % n], off_verts[i]]
        if len(set(vs)) < 4:
            continue
        try:
            q = bm.faces.new(vs)
        except ValueError as ex:
            print(f"[PolyOffset Notice] Ring face creation warning: {ex}")
            continue
        q.normal_update()
        if q.normal.dot(ref_n) < 0:
            q.normal_flip()
        ring.append(q)
    return [q for q in ring if is_valid_elem(q)]

def commit_polyline_offset(bm, orig_indices, world_poly, plane_origin_w, plane_normal_w,
                           plane_u, plane_v, mw, imw, local_tol, is_closed, dist):
    local_origin = imw @ plane_origin_w
    local_normal = (imw.to_3x3() @ plane_normal_w).normalized()
    local_u = (imw.to_3x3() @ plane_u).normalized()
    local_v = (imw.to_3x3() @ plane_v).normalized()

    def to_2d(p3d):
        v = p3d - local_origin
        return mathutils.Vector((v.dot(local_u), v.dot(local_v)))

    def to_3d(p2d):
        return local_origin + local_u * p2d.x + local_v * p2d.y

    poly2d = [to_2d(imw @ p) for p in world_poly]
    n = len(poly2d)

    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    orig_verts = [bm.verts[idx] for idx in orig_indices if 0 <= idx < len(bm.verts) and is_valid_elem(bm.verts[idx])]
    orig_2d = [to_2d(v.co) for v in orig_verts]

    coplanar_faces = []
    for f in bm.faces:
        if not is_valid_elem(f):
            continue
        if abs(f.normal.dot(local_normal)) < 0.999:
            continue
        if all(abs((v.co - local_origin).dot(local_normal)) < local_tol * 10.0 for v in f.verts):
            coplanar_faces.append(f)

    vert_map = {}
    created_verts = []

    # Static KDTree mapping to stored vertex objects
    kd_objs = []
    kd = mathutils.kdtree.KDTree(max(1, len(bm.verts)))
    for v in bm.verts:
        if is_valid_elem(v):
            kd.insert(v.co, len(kd_objs))
            kd_objs.append(v)
    kd.balance()

    # Dynamic spatial hash for new vertices created during split
    cell = max(local_tol, 1e-9)
    new_cells = {}

    def _ck(p):
        return (math.floor(p.x / cell), math.floor(p.y / cell), math.floor(p.z / cell))

    def _remember(v):
        new_cells.setdefault(_ck(v.co), []).append(v)

    def _find_new(p3d):
        cx, cy, cz = _ck(p3d)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for v in new_cells.get((cx + dx, cy + dy, cz + dz), ()):
                        if is_valid_elem(v) and (v.co - p3d).length < local_tol:
                            return v
        return None

    def register_vert(v):
        key = (round(to_2d(v.co).x, 4), round(to_2d(v.co).y, 4))
        vert_map[key] = v
        _remember(v)
        return v

    def get_or_create_vert(p2d):
        key = (round(p2d.x, 4), round(p2d.y, 4))
        v = vert_map.get(key)
        if v is not None and is_valid_elem(v):
            return v
        p3d = to_3d(p2d)
        v = _find_new(p3d)
        if v is None and kd_objs:
            _co, i, d = kd.find(p3d)
            if d <= local_tol and is_valid_elem(kd_objs[i]):
                v = kd_objs[i]
        if v is None:
            v = bm.verts.new(p3d)
            created_verts.append(v)
            _remember(v)
        vert_map[key] = v
        return v

    if is_closed:
        orig_set = set(orig_verts)
        target_f = next((f for f in coplanar_faces if is_valid_elem(f) and set(f.verts) == orig_set), None)
        if target_f:
            off_verts = [get_or_create_vert(p) for p in poly2d]
            if dist < 0:
                inner_f, ring_quads = replace_face_with_ring(bm, target_f, orig_verts, off_verts)
                return loop_edges(bm, off_verts, True), ""
            elif face_is_isolated(target_f):
                ring_quads = add_ring_only(bm, target_f, orig_verts, off_verts)
                return loop_edges(bm, off_verts, True), ""

    seg_pairs = []
    num_segs = n if is_closed else n - 1
    for i in range(num_segs):
        seg_pairs.append((poly2d[i], poly2d[(i + 1) % n]))

    if not is_closed and len(orig_2d) == n:
        seg_pairs.append((orig_2d[0], poly2d[0]))
        seg_pairs.append((orig_2d[-1], poly2d[-1]))

    cand_edges = list({e for f in coplanar_faces for e in f.edges if is_valid_elem(e)})
    seg_splits = [[(0.0, p1), (1.0, p2)] for p1, p2 in seg_pairs]

    for idx_sp, (p1, p2) in enumerate(seg_pairs):
        splits = seg_splits[idx_sp]
        seg_dir = p2 - p1
        seg_len2 = seg_dir.length_squared
        if seg_len2 < 1e-10:
            continue

        edges_to_check = list(cand_edges)
        for e in edges_to_check:
            if not is_valid_elem(e):
                continue
            e1, e2 = to_2d(e.verts[0].co), to_2d(e.verts[1].co)
            hit = mathutils.geometry.intersect_line_line_2d(p1, p2, e1, e2)
            if hit is None:
                continue
            t_seg = (hit - p1).dot(seg_dir) / seg_len2
            e_dir = e2 - e1
            e_len2 = e_dir.length_squared
            if e_len2 < 1e-10:
                continue
            t_edge = (hit - e1).dot(e_dir) / e_len2

            eps = 1e-4
            if -eps <= t_seg <= 1.0 + eps and eps < t_edge < 1.0 - eps:
                try:
                    e_new, v_split = bmesh.utils.edge_split(e, e.verts[0], max(1e-3, min(1.0 - 1e-3, t_edge)))
                    if is_valid_elem(e_new):
                        cand_edges.append(e_new)
                    if is_valid_elem(v_split):
                        register_vert(v_split)
                    splits.append((max(0.0, min(1.0, t_seg)), mathutils.Vector(hit)))
                except Exception as ex:
                    print(f"[PolyOffset Notice] Edge split exception: {ex}")

    new_edges = []
    path_edges = []
    for idx_sp, splits in enumerate(seg_splits):
        splits.sort(key=lambda item: item[0])
        prev_v = None
        for t, p in splits:
            v = get_or_create_vert(p)
            if prev_v is not None and prev_v != v:
                if (v.co - prev_v.co).length > local_tol * 0.5:
                    ne = bm.edges.get((prev_v, v)) or bm.edges.new((prev_v, v))
                    new_edges.append(ne)
                    if idx_sp < num_segs:
                        path_edges.append(ne)
            prev_v = v

    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    sliced_faces = []
    for f in coplanar_faces:
        if not is_valid_elem(f):
            continue
        poly_f = [to_2d(v.co) for v in f.verts]
        inside = []
        for e in new_edges:
            if not is_valid_elem(e) or e in f.edges:
                continue
            mid = (e.verts[0].co + e.verts[1].co) * 0.5
            if is_point_in_polygon_2d(to_2d(mid), poly_f):
                inside.append(e)
        if inside:
            try:
                res = bmesh.utils.face_split_edgenet(f, inside)
                split_f = res.get('faces', []) if isinstance(res, dict) else res
                for sf in split_f:
                    if is_valid_elem(sf):
                        sliced_faces.append(sf)
            except Exception as ex:
                print(f"[PolyOffset Notice] face_split_edgenet: {ex}")

    valid_created = [v for v in created_verts if is_valid_elem(v)]
    if valid_created:
        bmesh.ops.remove_doubles(bm, verts=valid_created, dist=local_tol)

    orphans = [e for e in new_edges if is_valid_elem(e) and not e.link_faces]
    warn_msg = ""
    if orphans:
        warn_msg = f"Poly Offset: {len(orphans)} offset edges did not split their face"

    return [e for e in path_edges if is_valid_elem(e)], warn_msg

def draw_callback_3d_offset(self, context):
    if bpy.context.area != self.area:
        return

    region, rv3d = get_3d_region_and_rv3d(bpy.context)
    if not region or not getattr(self, 'preview_chains', None):
        return

    gpu.state.blend_set('ALPHA')
    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float("viewportSize", (float(region.width), float(region.height)))

    for chain in self.preview_chains:
        orig = chain.get('orig_3d', [])
        off = chain.get('off_3d', [])
        is_closed = chain.get('closed', False)

        if orig:
            shader.uniform_float("lineWidth", 1.0)
            shader.uniform_float("color", (1.0, 1.0, 1.0, 0.35))
            d = list(orig)
            if is_closed and len(d) > 2:
                d.append(d[0])
            batch_for_shader(shader, 'LINE_STRIP', {"pos": d}).draw(shader)

        if off:
            shader.uniform_float("lineWidth", 3.0)
            shader.uniform_float("color", (1.0, 0.85, 0.1, 1.0))
            d = list(off)
            if is_closed and len(d) > 2:
                d.append(d[0])
            batch_for_shader(shader, 'LINE_STRIP', {"pos": d}).draw(shader)

            if not is_closed and len(orig) == len(off) and len(orig) >= 2:
                shader.uniform_float("lineWidth", 1.5)
                shader.uniform_float("color", (1.0, 0.85, 0.1, 0.6))
                cap1 = [orig[0], off[0]]
                cap2 = [orig[-1], off[-1]]
                batch_for_shader(shader, 'LINE_STRIP', {"pos": cap1}).draw(shader)
                batch_for_shader(shader, 'LINE_STRIP', {"pos": cap2}).draw(shader)

    gpu.state.blend_set('NONE')

def draw_callback_2d_offset(self, context):
    if bpy.context.area != self.area:
        return
    region, rv3d = get_3d_region_and_rv3d(bpy.context)
    if not region or not rv3d:
        return

    ui_scale = get_ui_scale(bpy.context)
    font_id = 0
    blf.size(font_id, int(13 * ui_scale))

    dist = getattr(self, 'offset_dist', 0.0)
    c0_closed = self.chains[0]['closed'] if getattr(self, 'chains', None) else True

    if c0_closed:
        side_lbl = "Inward" if getattr(self, 'is_inset_state', False) else "Outward"
        dist_str = f"{format_unit_length(bpy.context, abs(dist))} ({side_lbl})"
    else:
        dist_str = format_unit_length(bpy.context, abs(dist))

    clamp_lbl = "ON" if getattr(self, 'clamp_overlap', True) else "OFF"

    cx, cy = self.mouse_2d.x, self.mouse_2d.y
    blf.position(font_id, int(cx + 20 * ui_scale), int(cy + 12 * ui_scale), 0)
    if self.typed_buffer:
        blf.color(font_id, 1.0, 0.5, 0.0, 1.0)
        blf.draw(font_id, f"Distance: {self.typed_buffer} [ENTER]")
    else:
        blf.color(font_id, 0.95, 0.95, 0.95, 1.0)
        blf.draw(font_id, f"Offset: {dist_str} | Clamp: {clamp_lbl}")

    blf.position(font_id, 20, 40, 0)
    blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
    blf.draw(font_id,
             "Poly Offset v1.2.0: Move Mouse / Type Unit + Enter | Shift+F: Flip | Shift+C: Clamp | "
             "Enter/LMB: Confirm | Esc: Cancel")

    gpu.state.blend_set('NONE')

class MESH_OT_poly_offset(bpy.types.Operator):
    """CAD-grade exact offset of selected edges or face boundaries."""
    bl_idname = "mesh.poly_offset"
    bl_label = "Poly Offset"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.mode == 'EDIT_MESH'
                and context.edit_object is not None
                and context.edit_object.type == 'MESH')

    def invoke(self, context, event):
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)

        bm.normal_update()
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        mw = obj.matrix_world
        nm = mw.to_3x3().inverted().transposed()

        chains = collect_offset_chains(bm, mw)
        if not chains:
            self.report({'WARNING'}, "Select edges or a face to offset")
            return {'CANCELLED'}

        self.chains = []
        region, rv3d = get_3d_region_and_rv3d(context)

        for chain in chains:
            pts3d = chain['pts_3d']
            v_indices = chain['vert_indices']
            closed = chain['closed']

            adj_faces = chain_adjacent_faces(bm, v_indices, closed)
            face_cands = [((nm @ f.normal).normalized(), mw @ f.calc_center_median()) 
                          for f in adj_faces if is_valid_elem(f)]

            sel_adj_faces = [f for f in adj_faces if is_valid_elem(f) and f.select]

            if sel_adj_faces:
                init_normal = (nm @ sel_adj_faces[0].normal).normalized()
                init_origin = mw @ sel_adj_faces[0].calc_center_median()
            elif face_cands:
                rv3d_dir = (rv3d.view_matrix.to_3x3().inverted() @ mathutils.Vector((0, 0, -1))).normalized() if rv3d else mathutils.Vector((0, 0, 1))
                best_cand = max(face_cands, key=lambda c: abs(c[0].dot(rv3d_dir)))
                init_normal, init_origin = best_cand[0], best_cand[1]
            else:
                init_origin = sum(pts3d, mathutils.Vector()) / len(pts3d)
                init_normal = mathutils.Vector((0, 0, 1))

            init_u, init_v = compute_plane_basis(init_normal)

            extent = max((p - pts3d[0]).length for p in pts3d) if pts3d else 1.0
            plane_tol = 1e-4 * extent + 1e-6

            is_planar = True
            for p in pts3d:
                if abs((p - init_origin).dot(init_normal)) > plane_tol:
                    is_planar = False
                    break

            if not is_planar:
                self.report({'WARNING'}, "Poly Offset: selected chain is non-planar")

            self.chains.append({
                'vert_indices': v_indices,
                'pts_3d': pts3d,
                'closed': closed,
                'faces': face_cands,
                'plane_origin': init_origin,
                'plane_normal': init_normal,
                'plane_u': init_u,
                'plane_v': init_v,
                'is_planar': is_planar
            })

        self.area = context.area
        self.mouse_2d = mathutils.Vector((event.mouse_region_x, event.mouse_region_y))
        self.typed_buffer = ""
        self.side_flip = 1.0
        self.clamp_overlap = True
        self.offset_dist = 0.0
        self.is_inset_state = False
        self.preview_chains = []

        context.window.cursor_modal_set('NONE')
        args = (self, context)
        self._h3d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_3d_offset, args, 'WINDOW', 'POST_VIEW')
        self._h2d = bpy.types.SpaceView3D.draw_handler_add(draw_callback_2d_offset, args, 'WINDOW', 'POST_PIXEL')
        _active_draw_handlers.append((self._h3d, self._h2d))

        context.window_manager.modal_handler_add(self)
        tag_redraw(context)
        return {'RUNNING_MODAL'}

    def _to_2d(self, p3d, origin, u, v):
        vec = p3d - origin
        return mathutils.Vector((vec.dot(u), vec.dot(v)))

    def _to_3d(self, p2d, origin, u, v):
        return origin + u * p2d.x + v * p2d.y

    def _resolve_dist(self, context, mouse_signed):
        if self.typed_buffer:
            parsed = parse_unit_expression(context, self.typed_buffer)
            if parsed is not None and abs(parsed) > 1e-9:
                sign = 1.0 if mouse_signed >= 0 else -1.0
                return parsed * sign * self.side_flip
        return mouse_signed * self.side_flip

    def compute_previews(self, context):
        self.preview_chains = []
        self.is_inset_state = False
        raw = self.offset_dist

        for chain in self.chains:
            orig3 = chain['pts_3d']
            if len(orig3) < 2:
                continue

            orig2 = [self._to_2d(p, chain['plane_origin'], chain['plane_u'], chain['plane_v']) for p in orig3]
            dist = raw

            if chain['closed'] and dist < 0 and self.clamp_overlap:
                limit = calc_max_inward_clamp(orig2)
                if limit != float('inf'):
                    dist = -min(abs(dist), max(0.0, limit - 0.001))

            if chain['closed'] and dist < 0:
                self.is_inset_state = True

            off2 = offset_polyline_2d(orig2, dist, chain['closed'])
            off3 = [self._to_3d(p, chain['plane_origin'], chain['plane_u'], chain['plane_v']) for p in off2]

            self.preview_chains.append({
                'orig_3d': orig3,
                'off_3d': off3,
                'closed': chain['closed'],
                'vert_indices': chain['vert_indices'],
                'dist': dist,
                'plane_origin': chain['plane_origin'],
                'plane_normal': chain['plane_normal'],
                'plane_u': chain['plane_u'],
                'plane_v': chain['plane_v']
            })

    def modal(self, context, event):
        tag_redraw(context)

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                          'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE'}:
            return {'PASS_THROUGH'}

        region, rv3d = get_3d_region_and_rv3d(context)
        if not region or not rv3d:
            return {'PASS_THROUGH'}

        self.mouse_2d = mathutils.Vector((event.mouse_region_x, event.mouse_region_y))

        ray_o = view3d_utils.region_2d_to_origin_3d(region, rv3d, self.mouse_2d)
        ray_d = view3d_utils.region_2d_to_vector_3d(region, rv3d, self.mouse_2d).normalized()

        for ch in self.chains:
            if ch.get('faces'):
                pick = pick_face_under_cursor(ch['faces'], ch['pts_3d'], ray_o, ray_d)
                if pick:
                    ch['plane_origin'], ch['plane_normal'] = pick[0], pick[1]
                    ch['plane_u'], ch['plane_v'] = compute_plane_basis(ch['plane_normal'])

        c0 = self.chains[0]
        plane_hit = raycast_plane_intersection(
            region, rv3d, self.mouse_2d, c0['plane_origin'], c0['plane_normal']
        )
        mouse2 = self._to_2d(plane_hit, c0['plane_origin'], c0['plane_u'], c0['plane_v'])

        pts0_2 = [self._to_2d(p, c0['plane_origin'], c0['plane_u'], c0['plane_v']) for p in c0['pts_3d']]
        mouse_signed = signed_mouse_distance_2d(mouse2, pts0_2, c0['closed'])

        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            if self.typed_buffer:
                self.typed_buffer = ""
                self.offset_dist = self._resolve_dist(context, mouse_signed)
                self.compute_previews(context)
                return {'RUNNING_MODAL'}
            self.cleanup(context)
            return {'CANCELLED'}

        if event.value == 'PRESS':
            if event.type == 'F' and event.shift:
                self.side_flip *= -1.0
                self.offset_dist = self._resolve_dist(context, mouse_signed)
                self.compute_previews(context)
                return {'RUNNING_MODAL'}
            if event.type == 'C' and event.shift:
                self.clamp_overlap = not self.clamp_overlap
                self.offset_dist = self._resolve_dist(context, mouse_signed)
                self.compute_previews(context)
                return {'RUNNING_MODAL'}
            if event.type in {'Z', 'Y'} and (event.ctrl or event.oskey):
                if self.typed_buffer:
                    self.typed_buffer = ""
                    self.offset_dist = self._resolve_dist(context, mouse_signed)
                    self.compute_previews(context)
                return {'RUNNING_MODAL'}
            if event.type == 'BACK_SPACE':
                self.typed_buffer = self.typed_buffer[:-1]
                self.offset_dist = self._resolve_dist(context, mouse_signed)
                self.compute_previews(context)
                return {'RUNNING_MODAL'}
            if event.unicode and event.unicode in \
               "0123456789.+-*/'\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ":
                self.typed_buffer += event.unicode
                self.offset_dist = self._resolve_dist(context, mouse_signed)
                self.compute_previews(context)
                return {'RUNNING_MODAL'}

        self.offset_dist = self._resolve_dist(context, mouse_signed)
        self.compute_previews(context)

        if (event.type == 'LEFTMOUSE' and event.value == 'PRESS') or \
           (event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS'):
            try:
                self.commit(context)
            except Exception as ex:
                self.report({'ERROR'}, f"Poly Offset commit failed: {ex}")
                self.cleanup(context)
                return {'CANCELLED'}
            self.cleanup(context)
            return {'FINISHED'}

        return {'RUNNING_MODAL'}

    def commit(self, context):
        obj = context.edit_object
        bm = bmesh.from_edit_mesh(obj.data)

        bpy.ops.ed.undo_push(message="Poly Offset Commit")

        try:
            mw = obj.matrix_world
            imw = mw.inverted()

            obj_scale = max(mw.to_scale())
            obj_scale = obj_scale if obj_scale > 1e-6 else 1.0

            min_edge_len = float('inf')
            chain_extent = 0.0
            for ch in self.preview_chains:
                pts = ch['orig_3d']
                if pts:
                    ext = max((p - pts[0]).length for p in pts)
                    if ext > chain_extent:
                        chain_extent = ext
                for k in range(len(pts) - 1):
                    l = (pts[k+1] - pts[k]).length
                    if 0 < l < min_edge_len:
                        min_edge_len = l

            if min_edge_len == float('inf'):
                min_edge_len = 1.0

            local_tol = max(1e-3 * min_edge_len, 1e-6 * chain_extent) / obj_scale

            for v in bm.verts:
                if is_valid_elem(v): v.select = False
            for e in bm.edges:
                if is_valid_elem(e): e.select = False
            for f in bm.faces:
                if is_valid_elem(f): f.select = False

            selected_edges = []
            warnings = []

            for chain in self.preview_chains:
                off3 = chain['off_3d']
                orig_indices = chain['vert_indices']
                if len(off3) < 2:
                    continue

                results, warn_msg = commit_polyline_offset(
                    bm, orig_indices, off3, chain['plane_origin'], chain['plane_normal'],
                    chain['plane_u'], chain['plane_v'], mw, imw, local_tol, chain['closed'], chain['dist']
                )
                selected_edges.extend(results)
                if warn_msg:
                    warnings.append(warn_msg)

            if warnings:
                self.report({'WARNING'}, "; ".join(warnings))

            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            context.tool_settings.mesh_select_mode = (False, True, False)

            for e in selected_edges:
                if is_valid_elem(e):
                    e.select = True
                    for v in e.verts:
                        if is_valid_elem(v): v.select = True

            bm.select_flush_mode()
            bmesh.update_edit_mesh(obj.data)

        except Exception as ex:
            bpy.ops.ed.undo()
            self.report({'ERROR'}, f"Poly Offset commit failed and rolled back: {ex}")
            raise ex

    def cleanup(self, context):
        context.window.cursor_modal_restore()
        pair = (getattr(self, "_h3d", None), getattr(self, "_h2d", None))
        if pair in _active_draw_handlers:
            _active_draw_handlers.remove(pair)
        for h in pair:
            if h is not None:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
                except Exception as ex:
                    print(f"[PolyOffset Notice] Handler removal notice: {ex}")
        tag_redraw(context)


# ==============================================================================
# 5. UNIFIED UI PANEL & REGISTRATION
# ==============================================================================

class VIEW3D_PT_2d_sketch_panel(bpy.types.Panel):
    bl_label = "2D PolySketch"
    bl_idname = "VIEW3D_PT_2d_sketch_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Create'

    def draw(self, context):
        layout = self.layout
        
        col_sketch = layout.column(align=True)
        col_sketch.label(text="Object Mode Tools:")
        col_sketch.enabled = (context.mode == 'OBJECT')
        col_sketch.operator("mesh.poly_sketch_2d", text="Poly Sketch", icon='EDITMODE_HLT')
        
        layout.separator()
        
        col_edit = layout.column(align=True)
        col_edit.label(text="Edit Mode Tools:")
        col_edit.enabled = (context.mode == 'EDIT_MESH')
        col_edit.operator("mesh.poly_cut_2d", text="Poly Cut", icon='MESH_DATA')
        col_edit.operator("mesh.poly_offset", text="Poly Offset", icon='MOD_BEVEL')

def register():
    bpy.utils.register_class(MESH_OT_poly_sketch_2d)
    bpy.utils.register_class(MESH_OT_poly_cut_2d)
    bpy.utils.register_class(MESH_OT_poly_offset)
    bpy.utils.register_class(VIEW3D_PT_2d_sketch_panel)

def unregister():
    for h3d, h2d in list(_active_draw_handlers):
        if h3d: bpy.types.SpaceView3D.draw_handler_remove(h3d, 'WINDOW')
        if h2d: bpy.types.SpaceView3D.draw_handler_remove(h2d, 'WINDOW')
    _active_draw_handlers.clear()

    bpy.utils.unregister_class(VIEW3D_PT_2d_sketch_panel)
    bpy.utils.unregister_class(MESH_OT_poly_offset)
    bpy.utils.unregister_class(MESH_OT_poly_cut_2d)
    bpy.utils.unregister_class(MESH_OT_poly_sketch_2d)

if __name__ == "__main__":
    register()
