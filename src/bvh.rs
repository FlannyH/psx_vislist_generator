use glam::Vec3;

use crate::{structs::{Vertex, Axis}, aabb::AABB};

#[repr(C)]
#[derive(Debug)]
pub struct BvhNode {
    pub bounds: AABB,    // 24 bytes
    pub left_first: i32, // 4 bytes - if leaf, specifies first primitive index, otherwise, specifies node offset
    pub count: i32,      // 4 bytes - if non-zero, this is a leaf node
}
pub struct Bvh {
    pub nodes: Vec<BvhNode>, // node 0 is always the root node
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl Bvh {
    pub fn new(vertices: Vec<Vertex>) -> Self {        // Create new BVH
        let mut new_bvh = Self {
            nodes: Vec::new(),
            indices: (0..vertices.len() as u32).collect(),
            vertices,
        };

        // Create root node
        new_bvh.nodes.push(BvhNode {
            bounds: AABB::new(),
            left_first: 0,
            count: new_bvh.vertices.len() as _,
        });

        // Recursively break down into smaller nodes
        new_bvh.subdivide(0, 0);

        return new_bvh;
    }
    fn subdivide(&mut self, node_index: usize, rec_depth: usize) {
        // Get node
        let left = self.nodes.len();
        let node = &mut self.nodes[node_index];

        // Calculate node bounds
        let begin = node.left_first;
        let end = begin + node.count;
        for i in begin..end {
            let vertex = self
                .vertices
                .get(*self.indices.get(i as usize).unwrap() as usize)
                .unwrap();
            node.bounds.grow(vertex.position());
        }

        // Only subdivide if we have more than 2 triangles
        if node.count <= 40 || rec_depth > 6 {
            return;
        }

        // Get the average position of all the primitives
        let mut avg = Vec3::ZERO;
        let mut divide = 0;
        for i in begin..end {
            let vertex = self
                .vertices
                .get(*self.indices.get(i as usize).unwrap() as usize)
                .unwrap();
            avg += vertex.position();
            divide += 1;
        }
        avg /= divide as f32;

        // Determine split axis - choose biggest axis
        let size = node.bounds.max - node.bounds.min;
        let (split_axis, split_pos) = {
            if size.x > size.y && size.x > size.z {
                (Axis::X, avg.x)
            } else if size.y > size.x && size.y > size.z {
                (Axis::Y, avg.y)
            } else {
                (Axis::Z, avg.z)
            }
        };

        // Partition the index array, and get the split position
        let start_index = node.left_first;
        let node_count = node.count;
        node.count = 0; // this is not a leaf node.
        node.left_first = left as _; // this node has to point to the 2 child nodes
        let split_index = self.partition(split_axis, split_pos, start_index, node_count);
        let node = &mut self.nodes[node_index];

        // Abort if one of the sides is empty
        if split_index - start_index == 0 || split_index - start_index == node_count {
            node.count = node_count;
            return;
        }

        // Create 2 child nodes
        self.nodes.push(BvhNode {
            bounds: AABB::new(),
            left_first: start_index,
            count: split_index - start_index,
        });
        let right = self.nodes.len();
        self.nodes.push(BvhNode {
            bounds: AABB::new(),
            left_first: split_index,
            count: start_index + node_count - split_index,
        });

        // Subdivide further
        self.subdivide(left, rec_depth + 1);
        self.subdivide(right, rec_depth + 1);
    }

    fn partition(&mut self, axis: Axis, pivot: f32, start: i32, count: i32) -> i32 {
        let mut i = start;
        let mut j = start + count - 1;
        while i <= j {
            // Get triangle center
            let tri = &self.vertices[self.indices[i as usize] as usize];
            let center = match &axis {
                Axis::X => tri.x as f32,
                Axis::Y => tri.y as f32,
                Axis::Z => tri.z as f32,
            };

            // If the current primitive's center's <axis>-component is greated than the pivot's <axis>-component
            if center > pivot {
                (self.indices[i as usize], self.indices[j as usize]) =
                    (self.indices[j as usize], self.indices[i as usize]);
                j -= 1;
            } else {
                i += 1;
            }
        }

        return i;
    }

    pub fn pad_bounding_boxes(&mut self, x: f32, y: f32, z: f32) {
        for node in &mut self.nodes {
            node.bounds.max += Vec3::new(x, y, z);
            node.bounds.min -= Vec3::new(x, y, z);
        }
    }

}