---
icon: octicons/code-16
---

# API Reference

Here you can find detailed documentation for the public Python API of the `lqg` package.

## Core Systems & LQG Specification

::: lqg.system.System
    options:
      members:
        - simulate
        - simulate_actions
        - conditional_moments
        - conditional_distribution
        - log_likelihood
        - belief_tracking_distribution
        - to_numpyro

::: lqg.system.LQG
    options:
      members:
        - __init__

::: lqg.system.Dynamics

::: lqg.system.Actor

::: lqg.spec.LQGSpec

## Tracking Models

::: lqg.tracking.basic.TrackingTask

::: lqg.tracking.basic.BoundedActor

::: lqg.tracking.basic.OptimalActor

::: lqg.tracking.basic.RelativeObservationBoundedActor

::: lqg.tracking.subjective.SubjectiveActor

::: lqg.tracking.point_mass.PointMassBoundedActor

::: lqg.tracking.point_mass.PointMassSubjectiveActor

## Control & Estimation

::: lqg.control.lqr

::: lqg.belief.kf

## Inference & Models

::: lqg.infer.utils.infer

::: lqg.infer.mle

::: lqg.infer.models

::: lqg.infer.prior

## Utilities & I/O

::: lqg.io

::: lqg.optim

::: lqg.utils
