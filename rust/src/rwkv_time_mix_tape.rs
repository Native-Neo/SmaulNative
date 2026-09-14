use ndarray::{Array1, Array2, Array4};

pub struct RwkvTimeMixTape {
    pub input: Array2<f32>,
    pub prev: Array1<f32>,
    pub xr: Array2<f32>,
    pub xw: Array2<f32>,
    pub xk: Array2<f32>,
    pub xv: Array2<f32>,
    pub xa: Array2<f32>,
    pub xg: Array2<f32>,
    pub r: Array2<f32>,
    pub w_hidden: Array2<f32>,
    pub g_decay: Array2<f32>,
    pub k: Array2<f32>,
    pub v_base: Array2<f32>,
    pub v_correction: Option<Array2<f32>>,
    pub v_gate: Option<Array2<f32>>,
    pub v: Array2<f32>,
    pub a_hidden: Array2<f32>,
    pub a: Array2<f32>,
    pub g_hidden: Array2<f32>,
    pub g: Array2<f32>,
    pub kk_pre_norm: Array2<f32>,
    pub kk: Array2<f32>,
    pub k_mod: Array2<f32>,
    pub w: Array2<f32>,
    pub state_initial: Array4<f32>,
    pub y: Array2<f32>,
    pub normalized: Array2<f32>,
    pub correction: Array2<f32>,
    pub gated: Array2<f32>,
    pub output: Array2<f32>,
    pub v_first: Array2<f32>,
}

impl RwkvTimeMixTape {
    pub fn new(
        input: Array2<f32>,
        prev: Array1<f32>,
        state_initial: Array4<f32>,
        w_hidden: usize,
        v_hidden: Option<usize>,
        g_hidden: usize,
    ) -> Self {
        let shape = input.raw_dim();
        let channels = input.ncols();
        let rows = input.nrows();
        let state_shape = state_initial.raw_dim();
        Self {
            input,
            prev,
            xr: Array2::zeros(shape.clone()),
            xw: Array2::zeros(shape.clone()),
            xk: Array2::zeros(shape.clone()),
            xv: Array2::zeros(shape.clone()),
            xa: Array2::zeros(shape.clone()),
            xg: Array2::zeros(shape.clone()),
            r: Array2::zeros(shape.clone()),
            w_hidden: Array2::zeros((rows, w_hidden)),
            g_decay: Array2::zeros(shape.clone()),
            k: Array2::zeros(shape.clone()),
            v_base: Array2::zeros(shape.clone()),
            v_correction: v_hidden.map(|n| Array2::zeros((rows, n))),
            v_gate: v_hidden.map(|_| Array2::zeros(shape.clone())),
            v: Array2::zeros(shape.clone()),
            a_hidden: Array2::zeros((rows, w_hidden)),
            a: Array2::zeros(shape.clone()),
            g_hidden: Array2::zeros((rows, g_hidden)),
            g: Array2::zeros(shape.clone()),
            kk_pre_norm: Array2::zeros(shape.clone()),
            kk: Array2::zeros(shape.clone()),
            k_mod: Array2::zeros(shape.clone()),
            w: Array2::zeros(shape.clone()),
            state_initial: Array4::zeros(state_shape),
            y: Array2::zeros(shape.clone()),
            normalized: Array2::zeros(shape.clone()),
            correction: Array2::zeros(shape.clone()),
            gated: Array2::zeros(shape.clone()),
            output: Array2::zeros(shape.clone()),
            v_first: Array2::zeros(shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Array4};

    #[test]
    fn tape_preserves_forward_shapes() {
        let tape = RwkvTimeMixTape::new(
            Array2::zeros((3, 8)),
            Array1::zeros(8),
            Array4::zeros((1, 2, 4, 4)),
            5,
            Some(6),
            7,
        );
        assert_eq!(tape.xr.dim(), (3, 8));
        assert_eq!(tape.w_hidden.dim(), (3, 5));
        assert_eq!(tape.v_correction.as_ref().unwrap().dim(), (3, 6));
        assert_eq!(tape.state_initial.dim(), (1, 2, 4, 4));
        assert_eq!(tape.output.dim(), (3, 8));
    }
}
