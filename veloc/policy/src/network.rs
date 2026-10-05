use crate::Result;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Network {
    domain: Vec<[f32; 2]>,
    transform: Transform,
    offset: Vec<f32>,
    scale: Vec<f32>,
    support: Support,
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Support {
    points: Vec<Vec<f32>>,
    max_squared_distance: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Transform {
    Identity,
    Log1p,
    SignedLog1p,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Activation {
    Linear,
    Relu,
    Tanh,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Layer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: Activation,
}

/// Allocated once per session. Width and depth come from the validated model.
#[derive(Default)]
pub(crate) struct Workspace {
    current: Vec<f32>,
    next: Vec<f32>,
}

impl Network {
    pub fn validate(&self, inputs: usize, actions: usize) -> Result<()> {
        if inputs == 0
            || self.offset.len() != inputs
            || self.scale.len() != inputs
            || self.domain.len() != inputs
            || self.domain.iter().any(|&[lo, hi]| {
                !lo.is_finite()
                    || !hi.is_finite()
                    || lo > hi
                    || matches!(self.transform, Transform::Log1p) && lo <= -1.0
            })
            || self.offset.iter().any(|x| !x.is_finite())
            || self.scale.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || self.layers.is_empty()
        {
            return Err("invalid model input or normalization".into());
        }
        if self.support.points.is_empty()
            || !self.support.max_squared_distance.is_finite()
            || self.support.max_squared_distance < 0.0
            || self
                .support
                .points
                .iter()
                .any(|point| point.len() != inputs || point.iter().any(|x| !x.is_finite()))
        {
            return Err("invalid model support domain".into());
        }
        let mut width = inputs;
        for layer in &self.layers {
            if layer.bias.is_empty()
                || layer.weights.len() != layer.bias.len()
                || layer
                    .weights
                    .iter()
                    .any(|row| row.len() != width || row.iter().any(|w| !w.is_finite()))
                || layer.bias.iter().any(|b| !b.is_finite())
            {
                return Err("invalid model layer".into());
            }
            width = layer.bias.len();
        }
        if width != actions {
            return Err("model output/action mismatch".into());
        }
        Ok(())
    }

    pub fn workspace(&self) -> Workspace {
        let width = self
            .layers
            .iter()
            .map(|l| l.bias.len())
            .chain([self.offset.len()])
            .max()
            .unwrap();
        Workspace {
            current: vec![0.0; width],
            next: vec![0.0; width],
        }
    }

    pub fn choose(&self, inputs: impl Iterator<Item = f32>, work: &mut Workspace) -> usize {
        for (i, (x, &[lo, hi])) in inputs.zip(&self.domain).enumerate() {
            if !x.is_finite() || x < lo || x > hi {
                return 0;
            }
            let x = match self.transform {
                Transform::Identity => x,
                Transform::Log1p => x.ln_1p(),
                Transform::SignedLog1p => x.signum() * x.abs().ln_1p(),
            };
            work.current[i] = ((x - self.offset[i]) * self.scale[i]).clamp(-16.0, 16.0);
        }
        let inputs = self.offset.len();
        if !self.support.points.iter().any(|point| {
            let distance: f32 = point
                .iter()
                .zip(&work.current)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            distance <= self.support.max_squared_distance * inputs as f32
        }) {
            return 0;
        }
        for layer in &self.layers {
            for (i, (row, &bias)) in layer.weights.iter().zip(&layer.bias).enumerate() {
                let sum = row
                    .iter()
                    .zip(&work.current)
                    .fold(bias, |sum, (&w, &x)| sum + w * x);
                if !sum.is_finite() {
                    return 0;
                }
                work.next[i] = match layer.activation {
                    Activation::Linear => sum,
                    Activation::Relu => sum.max(0.0),
                    Activation::Tanh => sum.tanh(),
                };
            }
            std::mem::swap(&mut work.current, &mut work.next);
        }
        (1..self.layers.last().unwrap().bias.len()).fold(0, |best, i| {
            if work.current[i] > work.current[best] {
                i
            } else {
                best
            }
        })
    }
}
