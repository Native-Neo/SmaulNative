// End-to-end RQT: quantized forward/backward, full-precision master updates.
use ndarray::Array1;
use smaul_native::model_backward::ModelTrainStep;
use smaul_native::qat;
use smaul_native::rwkv_model::{RwkvModel, RwkvModelConfig};
use smaul_native::training::{OptimizerKind, TrainStep};

fn config() -> RwkvModelConfig { RwkvModelConfig::new(24, 16, 2, 8) }

fn loss_after(bits: u8, steps: usize) -> (f32, f32) {
    let mut model = RwkvModel::new(config(), 4242);
    let tokens = [1usize, 5, 2, 7, 3, 9];
    let targets = Array1::from_vec(vec![5, 2, 7, 3, 9, 1]);
    let mut train = TrainStep::new_with_optimizer(&model, 3e-3, OptimizerKind::AdamW);
    let mut first = 0.0;
    let mut last = 0.0;
    for i in 0..steps {
        let masters = (bits > 0).then(|| qat::quantize_in_place(&mut model, bits).unwrap());
        let step = ModelTrainStep::run_scaled(&model, &tokens, &targets, 1.0);
        if let Some(m) = masters { qat::restore_masters(&mut model, m); }
        if i == 0 { first = step.loss; }
        last = step.loss;
        train.step(&mut model, &step.gradients);
    }
    (first, last)
}

#[test]
fn rqt_training_reduces_loss_through_quantized_weights() {
    let (first, last) = loss_after(4, 30);
    assert!(first.is_finite() && last.is_finite());
    assert!(last < first, "4-bit RQT did not learn: {first} -> {last}");
}

#[test]
fn rqt_masters_stay_off_the_quantization_grid() {
    let mut model = RwkvModel::new(config(), 11);
    let tokens = [1usize, 2, 3];
    let targets = Array1::from_vec(vec![2, 3, 4]);
    let mut train = TrainStep::new_with_optimizer(&model, 1e-4, OptimizerKind::AdamW);

    for _ in 0..5 {
        let masters = qat::quantize_in_place(&mut model, 2).unwrap();
        // inside the window every row really is on its own 2-bit grid: {-s, 0, +s}
        for row in model.head.weight.rows() {
            let peak = row.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            assert!(row.iter().all(|v| *v == 0.0 || (v.abs() - peak).abs() <= 1e-6 * peak.max(1.0)),
                "2-bit window row should only contain 0 and +-scale");
        }
        let step = ModelTrainStep::run_scaled(&model, &tokens, &targets, 1.0);
        qat::restore_masters(&mut model, masters);
        train.step(&mut model, &step.gradients);
    }

    // after restoring, the masters must hold values the 2-bit grid cannot represent
    let mut off_grid = 0usize;
    for row in model.head.weight.rows() {
        let peak = row.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        off_grid += row.iter().filter(|v| **v != 0.0 && (v.abs() - peak).abs() > 1e-6 * peak.max(1.0)).count();
    }
    assert!(off_grid > 0, "masters collapsed onto the 2-bit grid; RQT is not keeping full precision");
}

#[test]
fn rqt_rejects_unsupported_widths() {
    let mut model = RwkvModel::new(config(), 1);
    assert!(qat::quantize_in_place(&mut model, 5).is_err());
    for bits in [2u8, 3, 4, 8] {
        let masters = qat::quantize_in_place(&mut model, bits).unwrap();
        qat::restore_masters(&mut model, masters);
    }
}

#[test]
fn restoring_masters_is_exact() {
    let mut model = RwkvModel::new(config(), 77);
    let before = model.head.weight.clone();
    let k = model.rwkv_blocks[0].time_mix.key.clone();
    let masters = qat::quantize_in_place(&mut model, 3).unwrap();
    qat::restore_masters(&mut model, masters);
    assert_eq!(model.head.weight, before);
    assert_eq!(model.rwkv_blocks[0].time_mix.key, k);
}
