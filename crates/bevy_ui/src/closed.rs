//! UI subtrees that nothing can see, which the per-frame UI passes skip.
//!
//! A node is *closed* when nothing beneath it can be displayed or reach the
//! layout of anything displayed:
//! - its [`Node::display`] is [`Display::None`], or
//! - it is a UI root (no UI parent) whose [`Visibility`] is
//!   [`Visibility::Hidden`], and no node beneath it overrides that with
//!   [`Visibility::Visible`].
//!
//! [`update_closed_subtrees`] keeps [`ClosedSubtree`] on closed nodes. The
//! layout's geometry pass, its taffy child sync, clipping and the
//! [`UiStack`](crate::UiStack) then stop at a closed node: its descendants
//! keep the layout, clip and stack index they had, and are left out of the
//! stack, so neither picking nor focus visits them. The closed node itself is
//! still laid out, clipped and stacked.
//!
//! The frame a node closes, everything beneath it is still walked once, so
//! what it leaves behind is what `bevy_ui` would have written for it closed
//! (zero sizes and full clips beneath a [`Display::None`]). The frame it
//! opens, its subtree is synced, laid out, clipped and stacked in full before
//! anything draws it. A node moved under a closed one makes it walk once
//! more for the same reason.
//!
//! Closing is settled in [`UiSystems::Prepare`](crate::UiSystems::Prepare):
//! show or hide UI before it, as with any layout input.

use crate::{Display, Node};
use bevy_camera::visibility::Visibility;
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::{Entity, EntityHashSet},
    hierarchy::{ChildOf, Children},
    lifecycle::RemovedComponents,
    prelude::{Component, ReflectComponent, ReflectResource, Resource},
    query::{Changed, Has, With},
    system::{Commands, Query, Res, ResMut, SystemParam},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

/// On a UI node nothing beneath which can be displayed (see the
/// [module docs](self)). Kept by [`update_closed_subtrees`]: read it, do not
/// insert it.
#[derive(Component, Debug, Default, Clone, Copy, PartialEq, Eq, Reflect)]
#[component(storage = "SparseSet")]
#[reflect(Component, Default, Debug, Clone, PartialEq)]
pub struct ClosedSubtree;

/// Whether the UI passes skip closed subtrees. On by default.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Resource, Default, Debug, Clone, PartialEq)]
pub struct SkipClosedSubtrees(pub bool);

impl Default for SkipClosedSubtrees {
    fn default() -> Self {
        Self(true)
    }
}

/// This frame's closings and openings, for the passes after
/// [`UiSystems::Prepare`](crate::UiSystems::Prepare).
#[derive(Resource, Debug, Default)]
pub struct ClosedSubtreeChanges {
    /// Closed nodes whose subtrees are walked in full this frame: closed this
    /// frame, or given a node from elsewhere.
    settling: EntityHashSet,
    /// Nodes opened this frame: their subtrees are synced to taffy afresh.
    opened: EntityHashSet,
}

impl ClosedSubtreeChanges {
    /// Whether the passes stop beneath `entity` this frame, given whether it
    /// carries [`ClosedSubtree`].
    pub fn skips_below(&self, entity: Entity, closed: bool) -> bool {
        closed && !self.settling.contains(&entity) && !self.opened.contains(&entity)
    }

    /// Whether `entity`'s descendants are left out of the
    /// [`UiStack`](crate::UiStack) this frame: it is closed, even if it closed
    /// only this frame.
    pub fn unstacks_below(&self, entity: Entity, closed: bool) -> bool {
        closed && !self.opened.contains(&entity)
    }

    /// Whether `entity` opened this frame.
    pub fn opened(&self, entity: Entity) -> bool {
        self.opened.contains(&entity)
    }
}

/// What [`update_closed_subtrees`] reads.
#[derive(SystemParam)]
pub struct ClosedSubtreeInputs<'w, 's> {
    nodes: Query<
        'w,
        's,
        (
            &'static Node,
            Option<&'static Visibility>,
            Has<ClosedSubtree>,
        ),
    >,
    parents: Query<'w, 's, &'static ChildOf>,
    hierarchy: Query<'w, 's, &'static Children>,
    restyled: Query<'w, 's, Entity, (With<Node>, Changed<Node>)>,
    reshown: Query<'w, 's, (Entity, &'static Visibility), (With<Node>, Changed<Visibility>)>,
    moved: Query<'w, 's, Entity, (With<Node>, Changed<ChildOf>)>,
    unparented: RemovedComponents<'w, 's, ChildOf>,
    all: Query<'w, 's, Entity, With<Node>>,
    closed: Query<'w, 's, Entity, With<ClosedSubtree>>,
    settings: Res<'w, SkipClosedSubtrees>,
}

impl ClosedSubtreeInputs<'_, '_> {
    fn parent(&self, entity: Entity) -> Option<Entity> {
        self.parents.get(entity).ok().map(ChildOf::parent)
    }

    /// No UI node above it (ghost nodes and plain entities do not count).
    fn is_ui_root(&self, entity: Entity) -> bool {
        !core::iter::successors(self.parent(entity), |&at| self.parent(at))
            .any(|at| self.nodes.contains(at))
    }

    fn children(&self, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
        self.hierarchy
            .get(entity)
            .ok()
            .into_iter()
            .flat_map(|children| children.iter().copied())
    }

    /// Whether a node beneath the hidden root `root` is drawn anyway.
    fn shows_through(&self, root: Entity) -> bool {
        let mut stack: Vec<Entity> = self.children(root).collect();
        while let Some(entity) = stack.pop() {
            if let Ok((node, visibility, _)) = self.nodes.get(entity) {
                if node.display == Display::None {
                    continue;
                }
                if visibility == Some(&Visibility::Visible) {
                    return true;
                }
            }
            stack.extend(self.children(entity));
        }
        false
    }

    /// Whether `entity` closes its subtree. A hidden root that is closed
    /// already is not looked beneath again unless `rescan`.
    fn closes(&self, entity: Entity, rescan: bool) -> bool {
        let Ok((node, visibility, closed)) = self.nodes.get(entity) else {
            return false;
        };
        if node.display == Display::None {
            return true;
        }
        if visibility != Some(&Visibility::Hidden) || !self.is_ui_root(entity) {
            return false;
        }
        (closed && !rescan) || !self.shows_through(entity)
    }

    /// The closed nodes above `entity`.
    fn closed_ancestors(&self, entity: Entity) -> impl Iterator<Item = Entity> + '_ {
        core::iter::successors(self.parent(entity), |&at| self.parent(at))
            .filter(|&at| self.closed.contains(at))
    }
}

/// Keeps [`ClosedSubtree`] on the nodes that close their subtrees, and
/// records this frame's closings and openings in [`ClosedSubtreeChanges`].
pub fn update_closed_subtrees(
    mut inputs: ClosedSubtreeInputs,
    mut changes: ResMut<ClosedSubtreeChanges>,
    mut commands: Commands,
) {
    changes.settling.clear();
    changes.opened.clear();

    if !inputs.settings.0 {
        for entity in &inputs.closed {
            commands.entity(entity).remove::<ClosedSubtree>();
            changes.opened.insert(entity);
        }
        return;
    }

    let mut candidates: EntityHashSet = inputs.restyled.iter().collect();
    candidates.extend(inputs.moved.iter());
    candidates.extend(inputs.unparented.read());
    if inputs.settings.is_changed() {
        candidates.extend(inputs.all.iter());
    }
    // A node shown with `Visibility::Visible` beneath a closed hidden root
    // opens it.
    let mut rescan = EntityHashSet::default();
    for (entity, visibility) in &inputs.reshown {
        candidates.insert(entity);
        if *visibility == Visibility::Visible {
            let root = core::iter::successors(Some(entity), |&at| inputs.parent(at))
                .last()
                .unwrap_or(entity);
            rescan.insert(root);
            candidates.insert(root);
        }
    }

    for entity in candidates {
        let Ok((_, _, closed)) = inputs.nodes.get(entity) else {
            continue;
        };
        let closes = inputs.closes(entity, rescan.contains(&entity));
        if closes && !closed {
            commands.entity(entity).insert(ClosedSubtree);
            changes.settling.insert(entity);
        } else if !closes && closed {
            commands.entity(entity).remove::<ClosedSubtree>();
            changes.opened.insert(entity);
        }
    }

    // A closed node beneath one that opened may never have been walked
    // closed (it closed while the other hid it): walk it once now.
    let opened: Vec<Entity> = changes.opened.iter().copied().collect();
    for entity in opened {
        let mut stack: Vec<Entity> = inputs.children(entity).collect();
        while let Some(at) = stack.pop() {
            if inputs.closed.contains(at) {
                changes.settling.insert(at);
            }
            stack.extend(inputs.children(at));
        }
    }

    // A node moved beneath a closed one still holds the geometry, clip and
    // stack index it had where it was: walk those closed nodes once more.
    let moved: Vec<Entity> = inputs.moved.iter().collect();
    for entity in moved {
        let closed: Vec<Entity> = inputs.closed_ancestors(entity).collect();
        changes.settling.extend(closed);
    }
}

#[cfg(test)]
mod tests {
    use super::{ClosedSubtree, SkipClosedSubtrees};
    use crate::{
        layout::{clipping::update_clipping_system, ui_surface::UiSurface},
        prelude::*,
        stack::ui_stack_system,
        sync_font_size_to_em_size, ui_layout_system,
        update::propagate_ui_target_cameras,
        CalculatedClip, ClosedSubtreeChanges, UiStack,
    };
    use bevy_app::{App, HierarchyPropagatePlugin, PostUpdate, PropagateSet, TaskPoolPlugin};
    use bevy_camera::{
        visibility::Visibility, Camera, Camera2d, ComputedCameraValues, RenderTargetInfo, Viewport,
    };
    use bevy_ecs::prelude::*;
    use bevy_math::{UVec2, Vec2};
    use bevy_text::RemSize;
    use bevy_utils::prelude::default;

    fn app(skip: bool) -> App {
        let mut app = App::new();
        app.add_plugins(TaskPoolPlugin::default());
        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));
        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiRenderTargetInfo>::new(
            PostUpdate,
        ));
        app.init_resource::<UiScale>();
        app.init_resource::<UiSurface>();
        app.init_resource::<UiStack>();
        app.init_resource::<bevy_text::TextPipeline>();
        app.init_resource::<bevy_text::FontCx>();
        app.init_resource::<RemSize>();
        app.init_resource::<bevy_text::ScaleCx>();
        app.init_resource::<ClosedSubtreeChanges>();
        app.insert_resource(SkipClosedSubtrees(skip));
        app.add_systems(
            PostUpdate,
            (
                ApplyDeferred,
                super::update_closed_subtrees,
                ApplyDeferred,
                propagate_ui_target_cameras,
                sync_font_size_to_em_size,
                ui_layout_system,
                ui_stack_system,
                update_clipping_system,
                ApplyDeferred,
            )
                .chain(),
        );
        for set in [
            PropagateSet::<ComputedUiTargetCamera>::default().intern(),
            PropagateSet::<ComputedUiRenderTargetInfo>::default().intern(),
        ] {
            app.configure_sets(
                PostUpdate,
                set.after(propagate_ui_target_cameras)
                    .before(ui_layout_system),
            );
        }
        app.world_mut().spawn((
            Camera2d,
            Camera {
                computed: ComputedCameraValues {
                    target_info: Some(RenderTargetInfo {
                        physical_size: UVec2::new(800, 600),
                        scale_factor: 1.,
                    }),
                    ..default()
                },
                viewport: Some(Viewport {
                    physical_size: UVec2::new(800, 600),
                    ..default()
                }),
                ..default()
            },
        ));
        app
    }

    /// A screen: a column with a clipping panel holding two rows.
    #[derive(Clone, Copy)]
    struct Screen {
        root: Entity,
        panel: Entity,
        row: Entity,
        cell: Entity,
    }

    fn screen(world: &mut World, left: f32) -> Screen {
        let cell = world
            .spawn(Node {
                width: Val::Px(30.),
                height: Val::Px(20.),
                ..default()
            })
            .id();
        let row = world
            .spawn(Node {
                width: Val::Px(200.),
                height: Val::Px(40.),
                padding: UiRect::all(Val::Px(4.)),
                ..default()
            })
            .add_child(cell)
            .id();
        let other = world
            .spawn(Node {
                width: Val::Px(120.),
                height: Val::Px(40.),
                ..default()
            })
            .id();
        let panel = world
            .spawn(Node {
                flex_direction: FlexDirection::Column,
                width: Val::Px(150.),
                height: Val::Px(60.),
                overflow: Overflow::clip(),
                ..default()
            })
            .add_children(&[row, other])
            .id();
        let root = world
            .spawn(Node {
                position_type: PositionType::Absolute,
                left: Val::Px(left),
                width: Val::Px(400.),
                height: Val::Px(300.),
                ..default()
            })
            .add_child(panel)
            .id();
        Screen {
            root,
            panel,
            row,
            cell,
        }
    }

    fn geometry(world: &World, entity: Entity) -> (Vec2, Vec2, Option<CalculatedClip>) {
        (
            world.get::<UiGlobalTransform>(entity).unwrap().translation,
            world.get::<ComputedNode>(entity).unwrap().size(),
            world.get::<CalculatedClip>(entity).cloned(),
        )
    }

    fn stacked(world: &World, entity: Entity) -> bool {
        world.resource::<UiStack>().uinodes.contains(&entity)
    }

    fn closed(world: &World, entity: Entity) -> bool {
        world.get::<ClosedSubtree>(entity).is_some()
    }

    #[test]
    fn a_hidden_root_is_skipped_and_its_first_shown_frame_is_laid_out() {
        let mut app = app(true);
        let shown = screen(app.world_mut(), 0.);
        let hidden = screen(app.world_mut(), 0.);
        app.update();
        app.world_mut()
            .entity_mut(hidden.root)
            .insert(Visibility::Hidden);
        app.update();
        assert!(closed(app.world(), hidden.root));
        assert!(!closed(app.world(), shown.root));
        // The root is stacked, nothing beneath it: picking and focus walk the
        // stack.
        assert!(stacked(app.world(), hidden.root));
        assert!(!stacked(app.world(), hidden.panel));
        assert!(!stacked(app.world(), hidden.cell));
        assert!(stacked(app.world(), shown.cell));

        // Changed in both while hidden: the hidden one is not laid out.
        let was = geometry(app.world(), hidden.cell);
        for s in [shown, hidden] {
            app.world_mut().get_mut::<Node>(s.row).unwrap().padding = UiRect::all(Val::Px(9.));
        }
        app.update();
        app.update();
        assert_eq!(geometry(app.world(), hidden.cell), was);
        assert_ne!(geometry(app.world(), shown.cell), was);

        // Shown: laid out, clipped and stacked that very frame.
        *app.world_mut().get_mut::<Visibility>(hidden.root).unwrap() = Visibility::Inherited;
        app.update();
        assert!(!closed(app.world(), hidden.root));
        for (a, b) in [
            (shown.panel, hidden.panel),
            (shown.row, hidden.row),
            (shown.cell, hidden.cell),
        ] {
            assert_eq!(geometry(app.world(), a), geometry(app.world(), b));
            assert!(stacked(app.world(), b));
        }
    }

    #[test]
    fn a_collapsed_subtree_settles_to_nothing_then_opens_laid_out() {
        let mut app = app(true);
        let s = screen(app.world_mut(), 0.);
        let reference = screen(app.world_mut(), 0.);
        app.update();
        for at in [s.row, reference.row] {
            app.world_mut().get_mut::<Node>(at).unwrap().display = Display::None;
        }
        app.update();
        assert!(closed(app.world(), s.row));
        // The frame it closed, what is beneath it was walked: no size, and
        // fully clipped, exactly as bevy_ui leaves a collapsed subtree.
        let (_, size, clip) = geometry(app.world(), s.cell);
        assert_eq!(size, Vec2::ZERO);
        assert_eq!(clip, Some(CalculatedClip::FullyClipped));
        assert!(!stacked(app.world(), s.cell));

        // A node spawned deep in it while collapsed, then the whole lot shown.
        let late = app
            .world_mut()
            .spawn(Node {
                width: Val::Px(10.),
                height: Val::Px(10.),
                ..default()
            })
            .id();
        app.world_mut().entity_mut(s.cell).add_child(late);
        app.update();
        app.update();
        app.world_mut().get_mut::<Node>(s.row).unwrap().display = Display::Flex;
        app.update();
        let (_, size, _) = geometry(app.world(), late);
        assert_eq!(size, Vec2::splat(10.));
        assert!(stacked(app.world(), late));
        assert_ne!(geometry(app.world(), s.cell).1, Vec2::ZERO);
        assert!(stacked(app.world(), s.cell));
    }

    #[test]
    fn a_node_moved_beneath_a_closed_one_is_left_with_nothing() {
        let mut app = app(true);
        let s = screen(app.world_mut(), 0.);
        let other = screen(app.world_mut(), 0.);
        app.update();
        app.world_mut().get_mut::<Node>(s.row).unwrap().display = Display::None;
        app.update();
        app.update();
        app.world_mut().entity_mut(s.row).add_child(other.cell);
        app.update();
        let (_, size, clip) = geometry(app.world(), other.cell);
        assert_eq!(size, Vec2::ZERO);
        assert_eq!(clip, Some(CalculatedClip::FullyClipped));
        assert!(!stacked(app.world(), other.cell));
    }

    #[test]
    fn a_collapsed_node_closed_while_its_root_hid_is_walked_when_the_root_opens() {
        let mut app = app(true);
        let s = screen(app.world_mut(), 0.);
        app.update();
        app.world_mut()
            .entity_mut(s.root)
            .insert(Visibility::Hidden);
        app.update();
        app.world_mut().get_mut::<Node>(s.row).unwrap().display = Display::None;
        app.update();
        app.update();
        *app.world_mut().get_mut::<Visibility>(s.root).unwrap() = Visibility::Inherited;
        app.update();
        let (_, size, clip) = geometry(app.world(), s.cell);
        assert_eq!(size, Vec2::ZERO);
        assert_eq!(clip, Some(CalculatedClip::FullyClipped));
        assert!(stacked(app.world(), s.row));
        assert!(!stacked(app.world(), s.cell));
    }

    #[test]
    fn a_visible_node_beneath_a_hidden_root_keeps_it_open() {
        let mut app = app(true);
        let s = screen(app.world_mut(), 0.);
        app.update();
        app.world_mut()
            .entity_mut(s.cell)
            .insert(Visibility::Visible);
        app.world_mut()
            .entity_mut(s.root)
            .insert(Visibility::Hidden);
        app.update();
        assert!(!closed(app.world(), s.root));
        assert!(stacked(app.world(), s.cell));

        // Shown through only after the root closed: it opens again.
        *app.world_mut().get_mut::<Visibility>(s.cell).unwrap() = Visibility::Inherited;
        *app.world_mut().get_mut::<Visibility>(s.root).unwrap() = Visibility::Inherited;
        app.update();
        *app.world_mut().get_mut::<Visibility>(s.root).unwrap() = Visibility::Hidden;
        app.update();
        assert!(closed(app.world(), s.root));
        *app.world_mut().get_mut::<Visibility>(s.cell).unwrap() = Visibility::Visible;
        app.update();
        assert!(!closed(app.world(), s.root));
        assert!(stacked(app.world(), s.cell));
    }

    /// Everything displayed is laid out, clipped and stacked exactly as
    /// without skipping, through closings, openings and edits.
    #[test]
    fn what_is_displayed_is_unchanged() {
        let mut apps = [app(true), app(false)];
        let mut screens = Vec::new();
        for app in &mut apps {
            let s: Vec<Screen> = (0..4)
                .map(|i| screen(app.world_mut(), 50. * i as f32))
                .collect();
            app.update();
            screens = s;
        }
        let steps: [&dyn Fn(&mut World, &[Screen]); 6] = [
            &|w, s| {
                w.entity_mut(s[1].root).insert(Visibility::Hidden);
            },
            &|w, s| w.get_mut::<Node>(s[2].row).unwrap().display = Display::None,
            &|w, s| {
                for x in s {
                    w.get_mut::<Node>(x.row).unwrap().width = Val::Px(260.);
                }
            },
            &|w, s| {
                *w.get_mut::<Visibility>(s[1].root).unwrap() = Visibility::Inherited;
            },
            &|w, s| {
                w.entity_mut(s[3].root).insert(Visibility::Hidden);
                w.get_mut::<Node>(s[2].row).unwrap().display = Display::Flex;
            },
            &|w, s| {
                for x in s {
                    w.get_mut::<Node>(x.panel).unwrap().height = Val::Px(30.);
                }
            },
        ];
        for step in steps {
            for app in &mut apps {
                step(app.world_mut(), &screens);
                app.update();
            }
            let [skipping, reference] = &mut apps;
            let mut displayed = Vec::new();
            for s in &screens {
                if skipping.world().get::<Visibility>(s.root) == Some(&Visibility::Hidden) {
                    displayed.push(s.root);
                    continue;
                }
                displayed.extend([s.root, s.panel]);
                if skipping.world().get::<Node>(s.row).unwrap().display != Display::None {
                    displayed.extend([s.row, s.cell]);
                } else {
                    displayed.push(s.row);
                }
            }
            for &e in &displayed {
                assert_eq!(
                    geometry(skipping.world(), e),
                    geometry(reference.world(), e),
                    "{e}"
                );
            }
            let order = |world: &World| -> Vec<Entity> {
                world
                    .resource::<UiStack>()
                    .uinodes
                    .iter()
                    .copied()
                    .filter(|e| displayed.contains(e))
                    .collect()
            };
            assert_eq!(order(skipping.world()), order(reference.world()));
        }
    }
}
