//! LIR state markers. Pass scheduling and state transitions belong to codegen.

#[derive(Debug, Clone, Copy, Default)]
pub struct RawLir;

#[derive(Debug, Clone, Copy, Default)]
pub struct LegalizedLir;

#[derive(Debug, Clone, Copy, Default)]
pub struct PreIselPrepared;

#[derive(Debug, Clone, Copy, Default)]
pub struct SelectedLir;

#[derive(Debug, Clone, Copy, Default)]
pub struct PostIselOptimized;

#[derive(Debug, Clone, Copy, Default)]
pub struct RegAllocated;

#[derive(Debug, Clone, Copy, Default)]
pub struct PrologueEpilogueInserted;

/// 允许创建“尚未显式绑定 bank”的 vreg 的阶段。
pub trait AllowsUnbankedVRegAlloc {}

impl AllowsUnbankedVRegAlloc for RawLir {}
impl AllowsUnbankedVRegAlloc for LegalizedLir {}
