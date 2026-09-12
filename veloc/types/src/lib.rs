//! Encoding-independent concrete type facts, shared by runtime IRs and definition tools.
//! No modules, instruction containers, allocation or code generation.
#![no_std]

/// Callable environment contracts, not a CPS calling convention.
/// Lifetime and call multiplicity are distinct; these are the combinations
/// supported by the IR, not a claim that every owned closure must be one-shot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallableKind {
    /// Borrows the creating activation; reusable while that activation is alive.
    Local,
    /// Owns captures; must be called, transferred or explicitly dropped once.
    Owned,
    /// Reentrant immutable environment containing only duplicable captures.
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Scalar {
    Int(u32),
    Float(u32),
    Bool,
    Ptr,
}

impl Scalar {
    #[inline]
    pub const fn element_bits(self) -> Option<u32> {
        match self {
            Self::Int(bits) | Self::Float(bits) => Some(bits),
            Self::Bool => Some(1),
            Self::Ptr => None,
        }
    }
}

/// Minimum lanes and whether they are multiplied by runtime vscale.
pub type Shape = (u16, bool);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Scalar(Scalar),
    Vector { element: Scalar, shape: Shape },
    Callable(CallableKind),
}

/// Concrete facts needed by pure type queries, independent of a storage encoding.
/// Nominal identity (scalar codes and module-owned signature IDs) stays in the
/// owning IR. Equal facts do not imply interchangeable IR types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Type(Kind);

impl Type {
    #[inline]
    pub const fn scalar(element: Scalar) -> Self {
        Self(Kind::Scalar(element))
    }

    #[inline]
    pub const fn vector(element: Scalar, lanes: u16, scalable: bool) -> Option<Self> {
        if matches!(element, Scalar::Ptr) || lanes < 2 || !lanes.is_power_of_two() {
            return None;
        }
        Some(Self(Kind::Vector {
            element,
            shape: (lanes, scalable),
        }))
    }

    #[inline]
    pub const fn callable(kind: CallableKind) -> Self {
        Self(Kind::Callable(kind))
    }

    #[inline]
    pub const fn element_bits(self) -> Option<u32> {
        match self.0 {
            Kind::Scalar(element) | Kind::Vector { element, .. } => element.element_bits(),
            Kind::Callable(_) => None,
        }
    }

    #[inline]
    pub const fn lanes(self) -> Option<u32> {
        match self.0 {
            Kind::Scalar(_) => Some(1),
            Kind::Vector {
                shape: (lanes, _), ..
            } => Some(lanes as u32),
            Kind::Callable(_) => None,
        }
    }

    #[inline]
    pub const fn shape(self) -> Option<Shape> {
        match self.0 {
            Kind::Vector { shape, .. } => Some(shape),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_fixed(self) -> bool {
        matches!(
            self.0,
            Kind::Vector {
                shape: (_, false),
                ..
            }
        )
    }

    #[inline]
    pub const fn is_callable(self) -> bool {
        matches!(self.0, Kind::Callable(_))
    }
    #[inline]
    pub const fn is_owned(self) -> bool {
        matches!(self.0, Kind::Callable(CallableKind::Owned))
    }
    #[inline]
    pub const fn is_local(self) -> bool {
        matches!(self.0, Kind::Callable(CallableKind::Local))
    }
    #[inline]
    pub const fn is_shared(self) -> bool {
        matches!(self.0, Kind::Callable(CallableKind::Shared))
    }

    /// Logical size; pointers and callables have no target-independent width.
    #[inline]
    pub fn bit_size(self) -> Option<TypeBits> {
        let bits = self.element_bits()?.checked_mul(self.lanes()?)?;
        Some(if matches!(self.shape(), Some((_, true))) {
            TypeBits::Scalable { min_bits: bits }
        } else {
            TypeBits::Fixed(bits)
        })
    }

    /// Byte-addressed representation: boolean lanes occupy one byte each.
    /// This is not a target ABI layout or a packed predicate-register size.
    #[inline]
    pub fn storage_size(self) -> TypeSize {
        let Some(bytes) = self
            .element_bits()
            .and_then(|bits| bits.div_ceil(8).checked_mul(self.lanes()?))
        else {
            return TypeSize::TargetDependent;
        };
        if matches!(self.shape(), Some((_, true))) {
            TypeSize::Scalable { min_bytes: bytes }
        } else {
            TypeSize::Fixed(bytes)
        }
    }

    #[inline]
    pub fn min_bytes(self) -> Option<u32> {
        self.storage_size().min_bytes()
    }
}

/// Storage size in the IR's byte-addressed representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSize {
    Fixed(u32),
    Scalable { min_bytes: u32 },
    TargetDependent,
}

impl TypeSize {
    #[inline]
    pub const fn fixed_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) => Some(bytes),
            _ => None,
        }
    }
    #[inline]
    pub const fn min_bytes(self) -> Option<u32> {
        match self {
            Self::Fixed(bytes) | Self::Scalable { min_bytes: bytes } => Some(bytes),
            Self::TargetDependent => None,
        }
    }
}

/// Logical bits, preserving runtime scale. Equal minima do not imply equal sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeBits {
    Fixed(u32),
    Scalable { min_bits: u32 },
}

impl TypeBits {
    #[inline]
    pub const fn fixed_bits(self) -> Option<u32> {
        match self {
            Self::Fixed(bits) => Some(bits),
            _ => None,
        }
    }
    #[inline]
    pub const fn min_bits(self) -> u32 {
        match self {
            Self::Fixed(bits) | Self::Scalable { min_bits: bits } => bits,
        }
    }
}
