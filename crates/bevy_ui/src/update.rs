//! This module contains systems that update the UI when something changes

use crate::{
    experimental::{UiChildren, UiRootNodes},
    ComputedUiRenderTargetInfo, ComputedUiTargetCamera, DefaultUiCamera, UiScale, UiTargetCamera,
};

pub use crate::layout::clipping::update_clipping_system;

use bevy_app::Propagate;
use bevy_camera::Camera;
use bevy_ecs::{
    entity::Entity,
    query::{Or, With},
    system::{Commands, Query, Res},
};
use bevy_math::UVec2;

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
    current_query: Query<(
        Option<&Propagate<ComputedUiTargetCamera>>,
        Option<&Propagate<ComputedUiRenderTargetInfo>>,
    )>,
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

    // The camera the last root named, with what its target gives: roots
    // mostly share one camera.
    let mut resolved: Option<(Option<Entity>, ComputedUiRenderTargetInfo)> = None;

    for root_entity in ui_root_nodes.iter() {
        let camera = target_camera_query
            .get(root_entity)
            .ok()
            .map(UiTargetCamera::entity)
            .or(default_camera_entity);

        let target_info = match resolved {
            Some((resolved_camera, info)) if resolved_camera == camera => info,
            _ => {
                let (scale_factor, physical_size) = camera
                    .and_then(|camera| camera_query.get(camera).ok())
                    .map(|camera| {
                        (
                            camera.target_scaling_factor().unwrap_or(1.) * ui_scale.0,
                            camera.physical_viewport_size().unwrap_or(UVec2::ZERO),
                        )
                    })
                    .unwrap_or((1., UVec2::ZERO));
                let info = ComputedUiRenderTargetInfo {
                    scale_factor,
                    physical_size,
                };
                resolved = Some((camera, info));
                info
            }
        };

        // Only what differs is inserted. An insert of the value a root
        // already holds still marks it changed, and every changed root is
        // then propagated again: with nothing changed, each root cost two
        // queued inserts here, two more of `Inherited` and a visit to each
        // of its children, every frame.
        let (current_camera, current_info) = current_query.get(root_entity).unwrap_or((None, None));
        let target_camera = ComputedUiTargetCamera { camera };
        if current_camera.is_none_or(|current| current.0 != target_camera) {
            commands
                .entity(root_entity)
                .try_insert(Propagate(target_camera));
        }
        if current_info.is_none_or(|current| current.0 != target_info) {
            commands
                .entity(root_entity)
                .try_insert(Propagate(target_info));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::update::{propagate_ui_target_cameras, update_clipping_system};
    use crate::CalculatedClip;
    use crate::ComputedUiRenderTargetInfo;
    use crate::ComputedUiTargetCamera;
    use crate::FixedNode;
    use crate::IsDefaultUiCamera;
    use crate::Node;
    use crate::Overflow;
    use crate::OverrideClip;
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

    /// A root whose camera and target did not change is not inserted into
    /// again, so nothing is propagated from it a second time: every root
    /// used to be inserted into, and its children visited, every frame.
    #[test]
    fn an_unchanged_root_is_not_propagated_again() {
        use bevy_app::Propagate;
        use bevy_ecs::{entity::Entity, world::World};

        let mut app = setup_test_app();
        let world = app.world_mut();

        let camera = world
            .spawn((
                Camera2d,
                Camera {
                    computed: ComputedCameraValues {
                        target_info: Some(RenderTargetInfo {
                            physical_size: UVec2::new(1000, 500),
                            scale_factor: 2.,
                        }),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
            .id();
        let root = world.spawn(Node::default()).id();
        let child = world.spawn((Node::default(), ChildOf(root))).id();

        // When each of the four was last written.
        fn written(
            world: &World,
            root: Entity,
            child: Entity,
        ) -> [bevy_ecs::change_detection::Tick; 4] {
            let (root, child) = (world.entity(root), world.entity(child));
            [
                root.get_change_ticks::<Propagate<ComputedUiTargetCamera>>()
                    .unwrap()
                    .changed,
                root.get_change_ticks::<Propagate<ComputedUiRenderTargetInfo>>()
                    .unwrap()
                    .changed,
                child
                    .get_change_ticks::<ComputedUiTargetCamera>()
                    .unwrap()
                    .changed,
                child
                    .get_change_ticks::<ComputedUiRenderTargetInfo>()
                    .unwrap()
                    .changed,
            ]
        }

        app.update();
        app.update();
        let settled = written(app.world(), root, child);
        app.update();
        app.update();
        assert_eq!(
            written(app.world(), root, child),
            settled,
            "nothing changed, and a root or its child was written again"
        );

        // A target that does change still reaches the root and its child.
        app.world_mut()
            .get_mut::<Camera>(camera)
            .unwrap()
            .computed
            .target_info = Some(RenderTargetInfo {
            physical_size: UVec2::new(1000, 500),
            scale_factor: 3.,
        });
        app.update();
        app.update();
        for entity in [root, child] {
            assert_eq!(
                app.world()
                    .get::<ComputedUiRenderTargetInfo>(entity)
                    .unwrap()
                    .scale_factor,
                3.
            );
        }
        assert_ne!(written(app.world(), root, child)[1], settled[1]);
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
            ComputedUiTargetCamera {
                camera: Some(camera)
            }
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
                ComputedUiTargetCamera {
                    camera: Some(camera)
                }
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
    fn fixed_node_opens_new_clipping_context() {
        let mut app = App::new();
        app.add_systems(bevy_app::Update, update_clipping_system);

        let grandchild = app.world_mut().spawn(Node::default()).id();
        let child = app
            .world_mut()
            .spawn(Node::default())
            .add_child(grandchild)
            .id();
        app.world_mut()
            .spawn(Node {
                overflow: Overflow::clip(),
                ..default()
            })
            .add_child(child);

        app.update();
        assert_eq!(
            app.world()
                .get::<CalculatedClip>(grandchild)
                .unwrap()
                .rects()
                .unwrap()
                .len(),
            1
        );

        app.world_mut().entity_mut(child).insert(FixedNode);
        app.update();
        assert!(app.world().get::<CalculatedClip>(grandchild).is_none());

        app.world_mut().entity_mut(child).remove::<FixedNode>();
        app.update();
        assert_eq!(
            app.world()
                .get::<CalculatedClip>(grandchild)
                .unwrap()
                .rects()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn override_clip_opens_new_clipping_context() {
        let mut app = App::new();
        app.add_systems(bevy_app::Update, update_clipping_system);

        let grandchild = app.world_mut().spawn(Node::default()).id();
        let child = app
            .world_mut()
            .spawn((Node::default(), OverrideClip))
            .add_child(grandchild)
            .id();
        app.world_mut()
            .spawn(Node {
                overflow: Overflow::clip(),
                ..default()
            })
            .add_child(child);

        app.update();
        assert!(app.world().get::<CalculatedClip>(grandchild).is_none());
    }
}
