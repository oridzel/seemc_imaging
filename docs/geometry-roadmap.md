# Geometry and rastering roadmap

## Recommendation

Use a hybrid boundary representation behind one transport-facing interface:

1. exact analytic surfaces for controlled SEM test structures;
2. watertight triangle surfaces for arbitrary imported structures;
3. tetrahedral volume meshes only for a later problem that genuinely needs a
   volume discretization.

The electron transport samples a continuous free-flight distance and asks only
whether an interface lies before the collision. A surface representation
answers that question directly. Filling every solid with tetrahedra adds memory,
point-location, and element-traversal work without improving the basic
surface-crossing decision.

## Minimal geometry contract

The physics kernel should depend on a small geometry protocol, not on Blender,
Gmsh, Trimesh, or a particular acceleration library:

```python
class Geometry:
    def first_hit(self, origin, direction, max_distance, current_region):
        """Return the nearest interface before max_distance, or None."""

    def region_at(self, point):
        """Return the material/vacuum region containing point."""
```

A hit record should contain:

- exact distance and position;
- outward unit normal;
- surface ID;
- region/material on both sides;
- backend primitive or triangle ID for diagnostics.

This contract and the analytic `Plane` backend are implemented in version
0.2.0. Version 0.3.0 adds the nearest-hit union `Scene` and the analytic
`TrapezoidalLine`. `SurfaceHit.normal` is crossing-oriented: it points from
`region_from` to `region_to`. Version 0.4.0 adds the production raster driver,
finite-beam sampling, population maps, uncertainty maps, and landing-surface
fractions.

Transport then follows the already-validated planar rule:

1. sample a collision-free path;
2. query the nearest interface along that finite segment;
3. if no interface is closer, complete the path and scatter;
4. otherwise stop at the interface, apply transmission/reflection/refraction,
   and do not force a collision;
5. move by a scale-aware numerical tolerance into the selected side and draw a
   fresh free path.

## Authoring and execution tools

| Need | Preferred tool | Role |
| --- | --- | --- |
| Planes, spheres, cylinders, cones, boxes, trapezoidal lines, trenches | Native analytic primitives | Exact intersections and normals; ideal validation geometries |
| Dimensionally controlled parametric structures | CadQuery/OpenCascade or Gmsh/OpenCASCADE | Scripted CAD/BRep generation and STEP interchange |
| Free-form structures and visual assembly | Blender | Interactive surface authoring and triangle export |
| Mesh loading, validation, transforms, repair diagnostics | Trimesh | Watertightness, winding, proximity, scene utilities |
| High-throughput first-hit queries | Embree through a replaceable adapter | Accelerated ray/triangle intersection |
| Tetrahedral volume discretization | Gmsh or TetGen | Deferred option for complex volumetric material maps or field coupling |

Blender therefore remains useful, but it should be an authoring tool rather
than a transport dependency. For reproducible nanoscale linewidth structures,
a parameterized Python/CAD description will usually be easier to audit and vary.

## Development order

1. **Completed:** replace the hard-coded planar calculation with a `Plane`
   backend, local boundary normals, region transitions, and bitwise-identical
   seeded transport.
2. **Completed for the first structured specimen:** analytic infinite
   extruded trapezoidal line with exact top and sidewall intersections.
3. **Completed for same-material unions:** nearest-hit scene traversal that
   suppresses buried faces. Material-to-material interfaces remain deferred.
4. **Completed for raw emission images:** deterministic two-dimensional raster,
   Gaussian beam spot, pixel identifiers, population-resolved count/yield/SEM
   maps, and per-surface landing fractions.
5. Add sphere, box, cylinder, and finite-length trapezoid primitives.
6. Add physical dwell/current scaling and detector acceptance to the emitted
   populations.
7. Add a triangle-mesh backend and compare every simple mesh against its
   analytic equivalent, including grazing incidence.
8. Add trenches and finite structures for genuinely two-dimensional expected
   contrast; the current infinite line repeats its mean profile along y.
9. Add multi-material interfaces only after the single-material geometry and
   signal decomposition are validated.

## Geometry tests required before image work

- plane backend exactly reproduces the current planar model;
- analytic normals and first-hit distances have closed-form checks;
- scale invariance over the nanometre dimensions of interest;
- grazing, tangent, edge, and coincident-surface cases;
- no double crossings at shared edges or Boolean seams;
- analytic and triangulated versions agree as tessellation is refined;
- material transition and barrier energy accounting are conserved;
- raster translation changes only geometry inputs, not random-stream identity.
