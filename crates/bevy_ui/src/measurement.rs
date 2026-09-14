use bevy_ecs::{
    prelude::{Component, Query},
    reflect::ReflectComponent,
};
use bevy_math::Vec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_text::{ComputedTextBlock, FontCx};
use core::fmt::Formatter;

use crate::widget::ImageMeasure;

use crate::widget::TextMeasure;

use crate::{BoxSizing, LayoutContext, Node, UiRect, Val};

impl core::fmt::Debug for ContentSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContentSize").finish()
    }
}

/// The amount of space available to a UI node along one axis.
///
/// See <https://www.w3.org/TR/css-sizing-3/#available>.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AvailableSpace {
    /// The amount of space available is the specified number of physical pixels.
    Definite(f32),
    /// The amount of space available is indefinite and the node should be laid out under a min-content constraint.
    MinContent,
    /// The amount of space available is indefinite and the node should be laid out under a max-content constraint.
    MaxContent,
}

impl AvailableSpace {
    /// Returns true for definite values, else false.
    pub const fn is_definite(self) -> bool {
        matches!(self, AvailableSpace::Definite(_))
    }

    /// Converts to an `Option`: definite values become `Some(value)`, constraints become `None`.
    pub const fn into_option(self) -> Option<f32> {
        match self {
            AvailableSpace::Definite(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the definite value, or `default` if the space is a constraint.
    pub fn unwrap_or(self, default: f32) -> f32 {
        self.into_option().unwrap_or(default)
    }
}

/// A length from a node's style, as seen by a [`Measure`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MeasureLength {
    /// The length is decided by the layout algorithm.
    Auto,
    /// A length in physical pixels.
    Px(f32),
    /// A fraction of the size of the containing block, where `1.0` is 100%.
    Percent(f32),
}

impl MeasureLength {
    /// Converts a [`Val`] into physical pixels, keeping percentages relative.
    pub(crate) fn from_val(val: Val, context: &LayoutContext) -> Self {
        match val {
            Val::Auto => MeasureLength::Auto,
            Val::Percent(value) => MeasureLength::Percent(value / 100.),
            Val::Px(value) => MeasureLength::Px(context.scale_factor * value),
            Val::VMin(value) => {
                MeasureLength::Px(context.physical_size.min_element() * value / 100.)
            }
            Val::VMax(value) => {
                MeasureLength::Px(context.physical_size.max_element() * value / 100.)
            }
            Val::Vw(value) => MeasureLength::Px(context.physical_size.x * value / 100.),
            Val::Vh(value) => MeasureLength::Px(context.physical_size.y * value / 100.),
        }
    }

    /// Like [`MeasureLength::from_val`], for lengths that cannot be auto, such as padding and
    /// border widths. [`Val::Auto`] becomes zero.
    pub(crate) fn from_val_or_zero(val: Val, context: &LayoutContext) -> Self {
        match Self::from_val(val, context) {
            MeasureLength::Auto => MeasureLength::Px(0.),
            length => length,
        }
    }

    /// Resolves the length against the size of the containing block.
    ///
    /// Returns `None` if the length is auto, or a percentage of an unknown size.
    pub(crate) fn maybe_resolve(self, context: Option<f32>) -> Option<f32> {
        match self {
            MeasureLength::Auto => None,
            MeasureLength::Px(value) => Some(value),
            MeasureLength::Percent(fraction) => context.map(|dim| dim * fraction),
        }
    }

    /// Like [`MeasureLength::maybe_resolve`], but a length that cannot be resolved is zero.
    pub(crate) fn resolve_or_zero(self, context: Option<f32>) -> f32 {
        self.maybe_resolve(context).unwrap_or(0.0)
    }
}

/// The lengths of the four sides of a node's padding or border, as seen by a [`Measure`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MeasureRect {
    pub left: MeasureLength,
    pub right: MeasureLength,
    pub top: MeasureLength,
    pub bottom: MeasureLength,
}

impl MeasureRect {
    /// Every side is zero pixels.
    pub const ZERO: Self = Self {
        left: MeasureLength::Px(0.),
        right: MeasureLength::Px(0.),
        top: MeasureLength::Px(0.),
        bottom: MeasureLength::Px(0.),
    };

    fn from_ui_rect(rect: UiRect, context: &LayoutContext) -> Self {
        Self {
            left: MeasureLength::from_val_or_zero(rect.left, context),
            right: MeasureLength::from_val_or_zero(rect.right, context),
            top: MeasureLength::from_val_or_zero(rect.top, context),
            bottom: MeasureLength::from_val_or_zero(rect.bottom, context),
        }
    }

    /// Resolves the left and right sides against `width`, and the top and bottom against `height`.
    /// Sides that cannot be resolved are zero.
    pub(crate) fn resolve_or_zero(self, width: Option<f32>, height: Option<f32>) -> ResolvedRect {
        ResolvedRect {
            left: self.left.resolve_or_zero(width),
            right: self.right.resolve_or_zero(width),
            top: self.top.resolve_or_zero(height),
            bottom: self.bottom.resolve_or_zero(height),
        }
    }
}

/// The sides of a [`MeasureRect`] resolved to physical pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ResolvedRect {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

/// The parts of a node's style that a [`Measure`] can read, in physical pixels.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MeasureStyle {
    /// Whether the size constraints refer to the border box or the content box.
    pub box_sizing: BoxSizing,
    pub width: MeasureLength,
    pub height: MeasureLength,
    pub min_width: MeasureLength,
    pub min_height: MeasureLength,
    pub max_width: MeasureLength,
    pub max_height: MeasureLength,
    pub padding: MeasureRect,
    pub border: MeasureRect,
    /// The preferred ratio of width to height.
    pub aspect_ratio: Option<f32>,
}

impl MeasureStyle {
    pub const DEFAULT: Self = Self {
        box_sizing: BoxSizing::DEFAULT,
        width: MeasureLength::Auto,
        height: MeasureLength::Auto,
        min_width: MeasureLength::Auto,
        min_height: MeasureLength::Auto,
        max_width: MeasureLength::Auto,
        max_height: MeasureLength::Auto,
        padding: MeasureRect::ZERO,
        border: MeasureRect::ZERO,
        aspect_ratio: None,
    };

    /// Resolves the measured parts of `node` in the given layout context.
    pub fn from_node(node: &Node, context: &LayoutContext) -> Self {
        Self {
            box_sizing: node.box_sizing,
            width: MeasureLength::from_val(node.width, context),
            height: MeasureLength::from_val(node.height, context),
            min_width: MeasureLength::from_val(node.min_width, context),
            min_height: MeasureLength::from_val(node.min_height, context),
            max_width: MeasureLength::from_val(node.max_width, context),
            max_height: MeasureLength::from_val(node.max_height, context),
            padding: MeasureRect::from_ui_rect(node.padding, context),
            border: MeasureRect::from_ui_rect(node.border, context),
            aspect_ratio: node.aspect_ratio,
        }
    }
}

impl Default for MeasureStyle {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Clamps a size between optional bounds, ignoring the bounds that are `None`.
pub(crate) trait MaybeClamp {
    fn maybe_clamp(self, min: Option<f32>, max: Option<f32>) -> Self;
}

impl MaybeClamp for f32 {
    fn maybe_clamp(self, min: Option<f32>, max: Option<f32>) -> f32 {
        match (min, max) {
            (Some(min), Some(max)) => self.min(max).max(min),
            (None, Some(max)) => self.min(max),
            (Some(min), None) => self.max(min),
            (None, None) => self,
        }
    }
}

impl MaybeClamp for Option<f32> {
    fn maybe_clamp(self, min: Option<f32>, max: Option<f32>) -> Option<f32> {
        self.map(|value| value.maybe_clamp(min, max))
    }
}

/// If exactly one of `width` and `height` is known, computes the other from `aspect_ratio`
/// (width divided by height). Otherwise returns both unchanged.
pub(crate) fn maybe_apply_aspect_ratio(
    width: Option<f32>,
    height: Option<f32>,
    aspect_ratio: Option<f32>,
) -> (Option<f32>, Option<f32>) {
    match (aspect_ratio, width, height) {
        (Some(ratio), Some(width), None) => (Some(width), Some(width / ratio)),
        (Some(ratio), None, Some(height)) => (Some(height * ratio), Some(height)),
        _ => (width, height),
    }
}

/// Inputs provided to [`Measure::measure`].
pub struct MeasureArgs<'a> {
    /// The known width from the layout algorithm if this size is definite.
    pub known_width: Option<f32>,
    /// The known height from the layout algorithm if this size is definite.
    pub known_height: Option<f32>,
    /// The horizontal space available for this UI node.
    pub available_width: AvailableSpace,
    /// The vertical space available for this UI node.
    pub available_height: AvailableSpace,
    /// Parley font database, needed for text measurement.
    pub font_system: &'a mut FontCx,
    /// Text layout buffer used to compute intrinsic text size when required.
    pub buffer: Option<&'a mut ComputedTextBlock>,
    /// Resolved style for this node in physical pixels.
    pub style: &'a MeasureStyle,
}

#[derive(Copy, Clone)]
/// Resolved values for per-axis size constraints.
pub struct ResolvedAxis {
    /// Resolved minimum size along this axis.
    pub min: Option<f32>,
    /// Resolved preferred size along this axis.
    pub preferred: Option<f32>,
    /// Resolved maximum size along this axis.
    pub max: Option<f32>,
    /// Effective size along this axis after applying known size and min/max clamping.
    pub effective: Option<f32>,
}

fn resolve_axis(
    known_size: Option<f32>,
    available_space: AvailableSpace,
    min_dim: MeasureLength,
    size_dim: MeasureLength,
    max_dim: MeasureLength,
) -> ResolvedAxis {
    let available = available_space.into_option();
    let min = min_dim.maybe_resolve(available);
    let preferred = size_dim.maybe_resolve(available);
    let max = max_dim.maybe_resolve(available);
    ResolvedAxis {
        min,
        preferred,
        max,
        effective: known_size.or(preferred.or(min).maybe_clamp(min, max)),
    }
}

impl MeasureArgs<'_> {
    /// Resolve the node's width constraints and the effective width.
    pub fn resolve_width(&self) -> ResolvedAxis {
        resolve_axis(
            self.known_width,
            self.available_width,
            self.style.min_width,
            self.style.width,
            self.style.max_width,
        )
    }

    /// Resolve the node's height constraints and the effective height.
    pub fn resolve_height(&self) -> ResolvedAxis {
        resolve_axis(
            self.known_height,
            self.available_height,
            self.style.min_height,
            self.style.height,
            self.style.max_height,
        )
    }
}

/// A `Measure` is used to compute the size of a ui node
/// when the size of that node is based on its content.
pub trait Measure: Send + Sync + 'static {
    /// Calculate the size of the node given the constraints.
    fn measure(&mut self, measure_args: MeasureArgs<'_>) -> Vec2;
}

/// A type to serve as Taffy's node context (which allows the content size of leaf nodes to be computed)
///
/// It has specific variants for common built-in types to avoid making them opaque and needing to box them
/// by wrapping them in a closure and a Custom variant that allows arbitrary measurement closures if required.
pub enum NodeMeasure {
    Fixed(FixedMeasure),
    Text(TextMeasure),
    Image(ImageMeasure),
    Custom(Box<dyn Measure>),
}

impl Measure for NodeMeasure {
    fn measure(&mut self, measure_args: MeasureArgs) -> Vec2 {
        match self {
            NodeMeasure::Fixed(fixed) => fixed.measure(measure_args),
            NodeMeasure::Text(text) => text.measure(measure_args),
            NodeMeasure::Image(image) => image.measure(measure_args),
            NodeMeasure::Custom(custom) => custom.measure(measure_args),
        }
    }
}

/// Returns the text layout buffer of the entity measured by `measure`, if it is a
/// [`NodeMeasure::Text`] and `needs_buffer` is true.
///
/// See [`TextMeasure::needs_buffer`].
pub fn text_measure_buffer<'a>(
    needs_buffer: bool,
    measure: &NodeMeasure,
    query: &'a mut Query<&mut ComputedTextBlock>,
) -> Option<&'a mut ComputedTextBlock> {
    // We avoid a query lookup whenever the buffer is not required.
    if !needs_buffer {
        return None;
    }
    let NodeMeasure::Text(TextMeasure { info }) = measure else {
        return None;
    };
    let Ok(computed) = query.get_mut(info.entity) else {
        return None;
    };
    Some(computed.into_inner())
}

/// A `FixedMeasure` is a `Measure` that ignores all constraints and
/// always returns the same size.
#[derive(Default, Clone)]
pub struct FixedMeasure {
    pub size: Vec2,
}

impl Measure for FixedMeasure {
    fn measure(&mut self, _: MeasureArgs) -> Vec2 {
        self.size
    }
}

/// A node with a `ContentSize` component is a node where its size
/// is based on its content.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct ContentSize {
    /// The `Measure` used to compute the intrinsic size
    #[reflect(ignore)]
    pub(crate) measure: Option<NodeMeasure>,
}

impl ContentSize {
    /// Set a `Measure` for the UI node entity with this component
    pub fn set(&mut self, measure: NodeMeasure) {
        self.measure = Some(measure);
    }

    /// Clear the current `Measure` for this UI node.
    pub fn clear(&mut self) {
        self.measure = None;
    }

    /// The `Measure` for this UI node, if one is set.
    ///
    /// The layout system takes the `Measure` when it next updates the node,
    /// so this is usually `None` after layout has run.
    pub fn measure(&self) -> Option<&NodeMeasure> {
        self.measure.as_ref()
    }

    /// Mutable access to the `Measure` for this UI node, if one is set.
    pub fn measure_mut(&mut self) -> Option<&mut NodeMeasure> {
        self.measure.as_mut()
    }

    /// Creates a `ContentSize` with a `Measure` that always returns given `size` argument, regardless of the UI layout's constraints.
    pub fn fixed_size(size: Vec2) -> ContentSize {
        let mut content_size = Self::default();
        content_size.set(NodeMeasure::Fixed(FixedMeasure { size }));
        content_size
    }
}
