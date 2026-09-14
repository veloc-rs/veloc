#![allow(non_camel_case_types)]
#[derive(Clone, Copy)]
enum Before { Add, Copy, Saddo, Missing }
#[derive(Clone, Copy)]
enum After { Add, Sub, Saddo }

// This independent host declares the contract expected by compiled value rules.
impl Before { const fn is_value_only(self) -> bool { !matches!(self, Self::Missing) } }
impl After { const fn is_value_only(self) -> bool { true } }

struct Machine {
    values: Vec<i32>,
    types: Vec<u8>,
}

impl Context for Machine {
    type Value = usize;
    type Type = u8;
    fn input(&self, index: usize) -> usize { index }
    fn result(&self, index: usize) -> usize { 2 + index }
    fn value_type(&self, value: usize) -> u8 { self.types[value] }
    fn temp(&mut self, ty: u8) -> usize {
        let id = self.values.len();
        self.values.push(0);
        self.types.push(ty);
        id
    }
    fn emit(&mut self, opcode: After, results: &[usize], inputs: &[usize]) {
        let a = self.values[inputs[0]];
        let b = self.values[inputs[1]];
        assert_eq!(self.types[results[0]], 32);
        match opcode {
            After::Add => self.values[results[0]] = a.wrapping_add(b),
            After::Sub => self.values[results[0]] = a.wrapping_sub(b),
            After::Saddo => {
                let (value, overflow) = a.overflowing_add(b);
                assert_eq!(self.types[results[1]], 1);
                self.values[results[0]] = value;
                self.values[results[1]] = i32::from(overflow);
            }
        }
    }
    fn bind(&mut self, result: usize, value: usize) {
        let dst = self.result(result);
        self.values[dst] = self.values[value];
    }
}

fn main() {
    for a in [i32::MIN, -123, -1, 0, 1, 42, i32::MAX] {
        for b in [i32::MIN, -17, -1, 0, 1, 91, i32::MAX] {
            let mut machine = Machine { values: vec![a, b, 0, 0], types: vec![32, 32, 32, 1] };
            assert!(lower(Before::Add, &mut machine));
            assert_eq!(machine.values[2], a.wrapping_add(b));
            assert!(lower(Before::Copy, &mut machine));
            assert_eq!(machine.values[2], a);
            assert!(lower(Before::Saddo, &mut machine));
            let (value, overflow) = a.overflowing_add(b);
            assert_eq!(machine.values[2], value);
            assert_eq!(machine.values[3], i32::from(overflow));
            let before = machine.values.clone();
            assert!(!lower(Before::Missing, &mut machine));
            assert_eq!(machine.values, before);
        }
    }
}
