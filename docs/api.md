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

::: lqg.system.LQG
    options:
      members:
        - __init__

::: lqg.system.Dynamics

::: lqg.system.Actor

::: lqg.spec.LQGSpec

## Control & Estimation

::: lqg.control.lqr

::: lqg.belief.kf

## Inference & Models

::: lqg.infer.mle

::: lqg.infer.models

::: lqg.infer.prior

## Utilities & I/O

::: lqg.io

::: lqg.optim

::: lqg.utils
