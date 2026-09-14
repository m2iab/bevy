//! This module contains systems that update the UI when something changes

use crate::{
    experimental::{UiChildren, UiRootNodes},
    ui_transform::UiGlobalTransform,
    CalculatedClip, ComputedUiRenderTargetInfo, ComputedUiTargetCamera, DefaultUiCamera, Display,
    Node, Outline, OverrideClip, UiScale, UiTargetCamera,
};

use super::ComputedNode;
use bevy_app::Propagate;
use bevy_camera::Camera;
use bevy_ecs::{
    change_detection::DetectChangesMut,
    entity::Entity,
    query::{Has, Or, With},
    system::{Commands, Query, Res},
};
use bevy_math::{Rect, UVec2};

/// Updates clipping for all nodes
pub fn update_clipping_system(
    mut commands: Commands,
    root_nodes: UiRootNodes,
    mut node_query: Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Has<OverrideClip>,
    )>,
    ui_children: UiChildren,
) {
    for root_node in root_nodes.iter() {
        update_clipping(
            &mut commands,
            &ui_children,
            &mut node_query,
            root_node,
            None,
        );
    }
}

fn update_clipping(
    commands: &mut Commands,
    ui_children: &UiChildren,
    node_query: &mut Query<(
        &Node,
        &ComputedNode,
        &UiGlobalTransform,
        Option<&mut CalculatedClip>,
        Has<OverrideClip>,
    )>,
    entity: Entity,
    mut maybe_inherited_clip: Option<Rect>,
) {
    let Ok((node, computed_node, transform, maybe_calculated_clip, has_override_clip)) =
        node_query.get_mut(entity)
    else {
        return;
    };

    // If the UI node entity has an `OverrideClip` component, discard any inherited clip rect
    if has_override_clip {
        maybe_inherited_clip = None;
    }

    // If `display` is None, clip the entire node and all its descendants by replacing the inherited clip with a default rect (which is empty)
    if node.display == Display::None {
        maybe_inherited_clip = Some(Rect::default());
    }

    // Update this node's CalculatedClip component
    if let Some(mut calculated_clip) = maybe_calculated_clip {
        if let Some(inherited_clip) = maybe_inherited_clip {
            // Replace the previous calculated clip with the inherited clipping rect
            if calculated_clip.clip != inherited_clip {
                *calculated_clip = CalculatedClip {
                    clip: inherited_clip,
                };
            }
        } else {
            // No inherited clipping rect, remove the component
            commands.entity(entity).remove::<CalculatedClip>();
        }
    } else if let Some(inherited_clip) = maybe_inherited_clip {
        // No previous calculated clip, add a new CalculatedClip component with the inherited clipping rect
        commands.entity(entity).try_insert(CalculatedClip {
            clip: inherited_clip,
        });
    }

    // Calculate new clip rectangle for children nodes
    let children_clip = if node.overflow.is_visible() {
        // The current node doesn't clip, propagate the optional inherited clipping rect to any children
        maybe_inherited_clip
    } else {
        // Find the current node's clipping rect and intersect it with the inherited clipping rect, if one exists
        // Content isn't clipped at the edges of the node but at the edges of the region specified by [`Node::overflow_clip_margin`].
        //
        // `clip_inset` should always fit inside `node_rect`.
        // Even if `clip_inset` were to overflow, we won't return a degenerate result as `Rect::intersect` will clamp the intersection, leaving it empty.
        let mut clip_rect =
            computed_node.resolve_clip_rect(node.overflow, node.overflow_clip_margin);
        clip_rect.min += transform.translation;
        clip_rect.max += transform.translation;
        Some(maybe_inherited_clip.map_or(clip_rect, |c| c.intersect(clip_rect)))
    };

    for child in ui_children.iter_ui_children(entity) {
        update_clipping(commands, ui_children, node_query, child, children_clip);
    }
}

/// Resolves [`ComputedNode::border_radius`], [`ComputedNode::outline_width`] and
/// [`ComputedNode::outline_offset`] against each node's computed size.
///
/// Runs in [`UiLayoutSystems::Resolve`](crate::UiLayoutSystems::Resolve), after the layout is computed.
/// These values don't trigger change detection on [`ComputedNode`].
pub fn update_border_radius_and_outline_system(
    mut node_query: Query<(
        &mut ComputedNode,
        &Node,
        Option<&Outline>,
        &ComputedUiRenderTargetInfo,
    )>,
) {
    for (mut node, style, maybe_outline, target) in &mut node_query {
        // Layout stores the reciprocal of the scale factor on each node and resolves lengths with
        // the reciprocal of that, so take the same round trip to get bit-identical results.
        let scale_factor = target.scale_factor().recip().recip();
        let target_size = target.physical_size().as_vec2();

        // We don't trigger change detection for changes to border radius
        node.bypass_change_detection().border_radius = style.border_radius.resolve(
            scale_factor,
            node.size,
            target_size,
        );

        if let Some(outline) = maybe_outline {
            // don't trigger change detection when only outlines are changed
            let node = node.bypass_change_detection();
            node.outline_width = if style.display != Display::None {
                outline
                    .width
                    .resolve(scale_factor, node.size().x, target_size)
                    .unwrap_or(0.)
                    .max(0.)
            } else {
                0.
            };

            node.outline_offset = outline
                .offset
                .resolve(scale_factor, node.size().x, target_size)
                .unwrap_or(0.)
                // Clamp outline offsets to at least the length of the node's shorter side
                // Negative offset outlines can be useful to create thing like in-set focus indicators
                .max(-0.5 * node.size.min_element());
        }
    }
}

pub fn propagate_ui_target_cameras(
    mut commands: Commands,
    default_ui_camera: DefaultUiCamera,
    ui_scale: Res<UiScale>,
    camera_query: Query<&Camera>,
    target_camera_query: Query<&UiTargetCamera>,
    ui_root_nodes: UiRootNodes,
    ui_children: UiChildren,
    propagate_query: Query<
        Entity,
        Or<(
            With<Propagate<ComputedUiTargetCamera>>,
            With<Propagate<ComputedUiRenderTargetInfo>>,
        )>,
    >,
) {
    let default_camera_entity = default_ui_camera.get();

    for entity in propagate_query.iter() {
        if ui_children.get_parent(entity).is_some() {
            commands.entity(entity).remove::<(
                Propagate<ComputedUiTargetCamera>,
                Propagate<ComputedUiRenderTargetInfo>,
            )>();
        }
    }

    for root_entity in ui_root_nodes.iter() {
        let camera = target_camera_query
            .get(root_entity)
            .ok()
            .map(UiTargetCamera::entity)
            .or(default_camera_entity)
            .unwrap_or(Entity::PLACEHOLDER);

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiTargetCamera { camera }));

        let (scale_factor, physical_size) = camera_query
            .get(camera)
            .ok()
            .map(|camera| {
                (
                    camera.target_scaling_factor().unwrap_or(1.) * ui_scale.0,
                    camera.physical_viewport_size().unwrap_or(UVec2::ZERO),
                )
            })
            .unwrap_or((1., UVec2::ZERO));

        commands
            .entity(root_entity)
            .try_insert(Propagate(ComputedUiRenderTargetInfo {
                scale_factor,
                physical_size,
            }));
    }
}

#[cfg(test)]
mod tests {
    use crate::update::propagate_ui_target_cameras;
    use crate::ComputedUiRenderTargetInfo;
    use crate::ComputedUiTargetCamera;
    use crate::IsDefaultUiCamera;
    use crate::Node;
    use crate::UiScale;
    use crate::UiTargetCamera;
    use bevy_app::App;
    use bevy_app::HierarchyPropagatePlugin;
    use bevy_app::PostUpdate;
    use bevy_app::PropagateSet;
    use bevy_camera::Camera;
    use bevy_camera::Camera2d;
    use bevy_camera::ComputedCameraValues;
    use bevy_camera::RenderTargetInfo;
    use bevy_ecs::hierarchy::ChildOf;
    use bevy_math::UVec2;
    use bevy_utils::default;

    fn setup_test_app() -> App {
        let mut app = App::new();

        app.init_resource::<UiScale>();

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiTargetCamera>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiTargetCamera>::default(),
        );

        app.add_plugins(HierarchyPropagatePlugin::<ComputedUiRenderTargetInfo>::new(
            PostUpdate,
        ));
        app.configure_sets(
            PostUpdate,
            PropagateSet::<ComputedUiRenderTargetInfo>::default(),
        );

        app.add_systems(bevy_app::Update, propagate_ui_target_cameras);

        app
    }

    #[test]
    fn update_context_for_single_ui_root() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale_factor = 10.;
        let physical_size = UVec2::new(1000, 500);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size,
                            scale_factor,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
            ComputedUiTargetCamera { camera }
        );

        assert_eq!(
            *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
            ComputedUiRenderTargetInfo {
                physical_size,
                scale_factor,
            }
        );
    }

    #[test]
    fn update_multiple_context_for_multiple_ui_roots() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode1a = world.spawn(Node::default()).id();
        let uinode2a = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2b = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2c = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode1b = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        for (uinode, camera, scale_factor, physical_size) in [
            (uinode1a, camera1, scale1, size1),
            (uinode1b, camera1, scale1, size1),
            (uinode2a, camera2, scale2, size2),
            (uinode2b, camera2, scale2, size2),
            (uinode2c, camera2, scale2, size2),
        ] {
            assert_eq!(
                *world.get::<ComputedUiTargetCamera>(uinode).unwrap(),
                ComputedUiTargetCamera { camera }
            );

            assert_eq!(
                *world.get::<ComputedUiRenderTargetInfo>(uinode).unwrap(),
                ComputedUiRenderTargetInfo {
                    physical_size,
                    scale_factor,
                }
            );
        }
    }

    #[test]
    fn update_context_on_changed_camera() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        world.entity_mut(uinode).insert(UiTargetCamera(camera2));

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );
    }

    #[test]
    fn update_context_after_parented() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let camera1 = world.spawn((Camera2d, IsDefaultUiCamera)).id();
        let camera2 = world.spawn(Camera2d).id();
        let parent = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let child = world.spawn(Node::default()).id();

        app.update();

        assert_eq!(
            app.world()
                .get::<ComputedUiTargetCamera>(child)
                .unwrap()
                .get(),
            Some(camera1)
        );

        app.world_mut().entity_mut(parent).add_child(child);
        app.update();

        assert_eq!(
            app.world()
                .get::<ComputedUiTargetCamera>(child)
                .unwrap()
                .get(),
            Some(camera2)
        );
    }

    #[test]
    fn update_context_after_parent_removed() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale1 = 1.;
        let size1 = UVec2::new(100, 100);
        let scale2 = 2.;
        let size2 = UVec2::new(200, 200);

        let camera1 = world
            .spawn((
                Camera2d,
                IsDefaultUiCamera,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size1,
                            scale_factor: scale1,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let camera2 = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size2,
                            scale_factor: scale2,
                        }),
                        ..Default::default()
                    },
                    ..default()
                },
            ))
            .id();

        // `UiTargetCamera` is ignored on non-root UI nodes
        let uinode1 = world.spawn((Node::default(), UiTargetCamera(camera2))).id();
        let uinode2 = world.spawn(Node::default()).add_child(uinode1).id();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale1
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );

        // Now `uinode1` is a root UI node its `UiTargetCamera` component will be used and its camera target set to `camera2`.
        world.entity_mut(uinode1).remove::<ChildOf>();

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .scale_factor(),
            scale2
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode1)
                .unwrap()
                .physical_size(),
            size2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode1)
                .unwrap()
                .get()
                .unwrap(),
            camera2
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode2)
                .unwrap()
                .get()
                .unwrap(),
            camera1
        );
    }

    #[test]
    fn update_great_grandchild() {
        let mut app = setup_test_app();
        let world = app.world_mut();

        let scale = 1.;
        let size = UVec2::new(100, 100);

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: size,
                            scale_factor: scale,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();

        let uinode = world.spawn(Node::default()).id();
        world.spawn(Node::default()).with_children(|builder| {
            builder.spawn(Node::default()).with_children(|builder| {
                builder.spawn(Node::default()).add_child(uinode);
            });
        });

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor,
            scale
        );

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .physical_size,
            size
        );

        assert_eq!(
            world
                .get::<ComputedUiTargetCamera>(uinode)
                .unwrap()
                .get()
                .unwrap(),
            camera
        );

        world.resource_mut::<UiScale>().0 = 2.;

        app.update();
        let world = app.world_mut();

        assert_eq!(
            world
                .get::<ComputedUiRenderTargetInfo>(uinode)
                .unwrap()
                .scale_factor(),
            2.
        );
    }

    #[test]
    fn border_radius_and_outline_resolve_against_computed_size() {
        use crate::update::update_border_radius_and_outline_system;
        use crate::{BorderRadius, ComputedNode, Display, Outline, ResolvedBorderRadius, Val};
        use bevy_color::Color;
        use bevy_ecs::{change_detection::DetectChanges, system::RunSystemOnce, world::World};
        use bevy_math::Vec2;

        let mut world = World::new();
        let target = ComputedUiRenderTargetInfo {
            scale_factor: 2.,
            physical_size: UVec2::new(800, 600),
        };
        let computed_node = ComputedNode {
            size: Vec2::new(200., 100.),
            ..Default::default()
        };
        let border_radius = BorderRadius {
            top_left: Val::Px(10.),
            top_right: Val::Percent(20.),
            bottom_right: Val::Vw(50.),
            bottom_left: Val::Auto,
        };

        let outlined = world
            .spawn((
                Node {
                    border_radius,
                    ..default()
                },
                computed_node,
                target,
                Outline::new(Val::Px(3.), Val::Percent(-80.), Color::WHITE),
            ))
            .id();
        let hidden = world
            .spawn((
                Node {
                    display: Display::None,
                    ..default()
                },
                computed_node,
                target,
                Outline::new(Val::Px(3.), Val::Vh(1.), Color::WHITE),
            ))
            .id();
        let without_outline = world
            .spawn((
                Node {
                    border_radius,
                    ..default()
                },
                computed_node,
                target,
            ))
            .id();

        world.clear_trackers();
        world
            .run_system_once(update_border_radius_and_outline_system)
            .unwrap();

        let resolved_radius = ResolvedBorderRadius {
            // 10px at a scale factor of 2
            top_left: 20.,
            // 20% of the shorter side
            top_right: 20.,
            // 50vw, clamped to half of the shorter side
            bottom_right: 50.,
            bottom_left: 0.,
        };

        let node = world.entity(outlined).get_ref::<ComputedNode>().unwrap();
        assert!(!node.is_changed());
        assert_eq!(node.border_radius, resolved_radius);
        assert_eq!(node.outline_width, 6.);
        // -80% of the width, clamped to half of the shorter side
        assert_eq!(node.outline_offset, -50.);

        let node = world.entity(hidden).get_ref::<ComputedNode>().unwrap();
        assert!(!node.is_changed());
        assert_eq!(node.border_radius, ResolvedBorderRadius::ZERO);
        assert_eq!(node.outline_width, 0.);
        // 1vh
        assert_eq!(node.outline_offset, 6.);

        let node = world.get::<ComputedNode>(without_outline).unwrap();
        assert_eq!(node.border_radius, resolved_radius);
        assert_eq!(node.outline_width, 0.);
        assert_eq!(node.outline_offset, 0.);
    }
}
