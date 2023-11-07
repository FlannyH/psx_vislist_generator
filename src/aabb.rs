use glam::Vec3;

#[repr(C)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new() -> AABB {
        AABB {
            min: Vec3 {
                x: f32::INFINITY,
                y: f32::INFINITY,
                z: f32::INFINITY,
            },
            max: Vec3 {
                x: -f32::INFINITY,
                y: -f32::INFINITY,
                z: -f32::INFINITY,
            },
        }
    }

    pub fn grow(&mut self, position: Vec3) {
        self.min.x = self.min.x.min(position.x);
        self.min.y = self.min.y.min(position.y);
        self.min.z = self.min.z.min(position.z);
        self.max.x = self.max.x.max(position.x);
        self.max.y = self.max.y.max(position.y);
        self.max.z = self.max.z.max(position.z);
    }

    pub fn _area(&mut self) -> f32 {
        let size = self.max - self.min;
        size.x * size.y + size.y * size.z + size.z * size.x
    }
}
