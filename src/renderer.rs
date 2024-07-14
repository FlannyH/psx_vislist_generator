use std::{ffi::c_void, mem::size_of, f32::consts::PI, collections::HashMap};

use gl::types::{GLenum, GLfloat, GLvoid};
use glam::{Vec3, Mat4, Vec4};
use glfw::{Glfw, Window, Context};
use memoffset::offset_of;

use crate::structs::Vertex;

const RESOLUTION: u32 = 128;

pub struct Renderer {
    program: u32,
    vao: u32,
    vbo: u32,
    n_vertices: i32,
    window: Window,
    glfw: Glfw,
}

impl Renderer {
pub fn new() -> Self {
    // Set up a basic OpenGL setup
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    // Create an invisible window
    glfw.window_hint(glfw::WindowHint::Visible(false));
    let (window, _events) = glfw
        .create_window(RESOLUTION, RESOLUTION, "title", glfw::WindowMode::Windowed)
        .expect("Failed to create window.");
    glfw.make_context_current(Some(&window));
    glfw.set_swap_interval(glfw::SwapInterval::None);

    // Init OpenGL
    gl::load_with(|f_name| glfw.get_proc_address_raw(f_name));
    unsafe {
        let error = gl::GetError();
        if error != gl::NO_ERROR {
            panic!();
        }
    }
    let program;

    // Create shader
    unsafe {
        program = gl::CreateProgram();
        Self::load_shader_part(
            gl::VERTEX_SHADER,
            String::from(
                "
                #version 460

                // Vertex input
                layout (location = 0) in vec3 i_position;
                layout (location = 1) in vec3 i_color;
                layout (location = 2) in float i_section_id;

                // View matrix
                layout (location = 0) uniform mat4 u_matrix;
                layout (location = 1) uniform vec3 u_position;

                // Vert output
                out float o_section_id;
                out vec3 o_position;
                out vec3 o_color;

                void main() {
                    gl_Position = u_matrix * vec4(i_position, 1);
                    o_position = i_position;
                    o_section_id = i_section_id;
                    o_color = i_color;
                    o_color.x /= 255.0;
                    o_color.y /= 255.0;
                    o_color.z /= 255.0;
                }
            ",
            ),
            program,
        );
        Self::load_shader_part(
            gl::FRAGMENT_SHADER,
            String::from(
                "
                #version 460

                in float o_section_id;
                in vec3 o_position;
                in vec3 o_color;
                out vec4 frag_color;

                void main() {
                    frag_color = vec4(o_section_id / 255.0, gl_FragCoord.w / 4.0, 0.0, 1.0);
                    frag_color = vec4(
                        o_section_id / 255.0, 
                        gl_FragCoord.z, 
                        abs(mod(o_position.y, 512.0) - 256), 
                        256.0) / vec4(256.0);
                    frag_color.w = o_section_id / 255.0;
                    frag_color.x = o_color.x;
                    frag_color.y = o_color.y;
                    frag_color.z = o_color.z;
                }
            ",
            ),
            program,
        );

        gl::LinkProgram(program);
        gl::UseProgram(program);
    }
    Self {
        program: program,
        vao: 0,
        vbo: 0,
        n_vertices: 0,
        window,
        glfw,
    }
}

pub fn load_shader_part(shader_type: GLenum, source: String, program: u32) {
    let source_len = source.len() as i32;

    unsafe {
        // Create shader part
        let shader = gl::CreateShader(shader_type);
        gl::ShaderSource(shader, 1, &source.as_bytes().as_ptr().cast(), &source_len);
        gl::CompileShader(shader);

        // Check for errors
        let mut result = 0;
        let mut log_length = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut result);
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut log_length);
        let mut error_message: Vec<u8> = vec![0; log_length as usize];
        gl::GetShaderInfoLog(
            shader,
            log_length,
            std::ptr::null_mut(),
            error_message.as_mut_ptr().cast(),
        );

        // Did we get an error?
        if log_length > 0 {
            println!(
                "Shader compilation error!\n{}",
                std::str::from_utf8(error_message.as_slice()).unwrap()
            )
        }

        // Attach to program
        gl::AttachShader(program, shader);
    }
}

pub fn upload_mesh_to_gpu(&mut self, vertices: &Vec<Vertex>) {
    // Upload the mesh to the GPU
    unsafe {
        // Generate buffers
        gl::GenVertexArrays(1, &mut self.vao);
        gl::GenBuffers(1, &mut self.vbo);
        gl::BindVertexArray(self.vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo);

        // Define vertex layout
        gl::VertexAttribPointer(
            0,
            3,
            gl::SHORT,
            gl::FALSE,
            size_of::<Vertex>() as i32,
            offset_of!(Vertex, x) as *const _,
        );
        gl::VertexAttribPointer(
            1,
            3,
            gl::UNSIGNED_BYTE,
            gl::FALSE,
            size_of::<Vertex>() as i32,
            offset_of!(Vertex, r) as *const _,
        );
        gl::VertexAttribPointer(
            2,
            1,
            gl::UNSIGNED_SHORT,
            gl::FALSE,
            size_of::<Vertex>() as i32,
            offset_of!(Vertex, section_id) as *const _,
        );
        gl::EnableVertexAttribArray(0);
        gl::EnableVertexAttribArray(1);
        gl::EnableVertexAttribArray(2);

        // Upload the buffer
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (size_of::<Vertex>() * vertices.len()) as isize,
            &vertices[0] as *const Vertex as *const c_void,
            gl::STATIC_DRAW,
        );
    }
    self.n_vertices = vertices.len() as _;
}

pub fn get_visibility_at_position(&mut self, position: Vec3, _dbg_curr_field: u128, _node_id: usize) -> u128 {    
    // Set uniform variables

    //let modified_position = Vec4::new(-Z)

    let mut view_matrices = [
        glam::mat4(Vec4::Z, Vec4::NEG_Y, Vec4::X, Vec4::W),
        glam::mat4(Vec4::NEG_Z, Vec4::NEG_Y, Vec4::NEG_X, Vec4::W),
        glam::mat4(Vec4::NEG_X, Vec4::Z, Vec4::Y, Vec4::W),
        glam::mat4(Vec4::NEG_X, Vec4::NEG_Z, Vec4::NEG_Y, Vec4::W),
        glam::mat4(Vec4::NEG_X, Vec4::NEG_Y, Vec4::Z, Vec4::W),
        glam::mat4(Vec4::X, Vec4::NEG_Y, Vec4::NEG_Z, Vec4::W),
    ];    
    for mat in &mut view_matrices {
        mat.w_axis = mat.mul_vec4((-position).extend(1.0));
    }
    let proj_matrix = Mat4::perspective_lh(PI / 4.0, 1.0, 0.01, 100000.0);
    let buffer_size = (RESOLUTION * RESOLUTION * 4) as usize;
    let mut vis_field = 0u128;
    let mut buffer = vec![0u8; buffer_size];

    for view_matrix in &view_matrices {
        let view_matrix = *view_matrix;
        let comb_mat = proj_matrix * view_matrix;
        unsafe {
            gl::UseProgram(self.program);
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            gl::UniformMatrix4fv(0, 1, gl::FALSE, comb_mat.as_ref().as_ptr() as *const GLfloat);
            gl::Uniform3fv(1, 1, position.as_ref().as_ptr() as *const GLfloat);
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::DrawArrays(gl::TRIANGLES, 0, self.n_vertices as i32);
            gl::ReadPixels(0, 0, RESOLUTION as i32, RESOLUTION as i32, gl::RGBA, gl::UNSIGNED_BYTE, buffer.as_mut_ptr() as *mut GLvoid);
            self.window.swap_buffers();
            self.glfw.poll_events();
        }
        let mut representations = HashMap::<u8, i32>::new();

        // Check each pixel in the cubemap side, and count how many pixels each section occupies in it
        for i in (3..buffer_size).step_by(4) {
            if buffer[i] != 255 {
                *representations.entry(buffer[i]).or_insert(0) += 1;
            }
        }

        // For each pair of section (key) and pixel representation (value)
        for (key, value) in representations {
            // Ignore if the section is not represented enough
            if (value as f64 / (RESOLUTION * RESOLUTION) as f64) <= 0.00025 {
                continue;
            }

            vis_field |= 1 << key;
        }
    }

    vis_field
}
}