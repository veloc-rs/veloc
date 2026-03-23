#[derive(Debug, Clone, Copy, Default)]
pub struct RawMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct LegalizedMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct PreIselPrepared;

#[derive(Debug, Clone, Copy, Default)]
pub struct SelectedMir;

#[derive(Debug, Clone, Copy, Default)]
pub struct PostIselOptimized;

#[derive(Debug, Clone, Copy, Default)]
pub struct RegAllocated;

#[derive(Debug, Clone, Copy, Default)]
pub struct PrologueEpilogueInserted;

/// 允许创建“尚未显式绑定 bank”的 vreg 的阶段。
pub trait AllowsUnbankedVRegAlloc {}

impl AllowsUnbankedVRegAlloc for RawMir {}
impl AllowsUnbankedVRegAlloc for LegalizedMir {}
