# Plan for internal evals

## Goal & motivation
We want to support evaluations and finetuning on various datasets.
Asparagus provides a way of doing that, but we've hit some limitations with it:
- It's third party code, and it has some bugs (which is ok)
- It's a dependency, and it also relies on other external dependencies (e.g. gardening tools)
- This level of abstraction can be a bit tough to follow
- The tasks that it supports rely on very small datasets, so are with limited quality. The only task with a decent amount of data is task 3, the brain age task.

So, we want to create our own, simple finetuning & eval suite to which we can add different tasks.

## Requirements
Put code in src/evaluation

We want to support the following fine-tuning modes:
- probe -> the backbone is frozen, only the head on top is trained
- full -> both the backbone and the head are trained. This should exist as a config, but shouldn't be implmented yet.

When passing the output of the backbone to the head, multiple representations can be specified:
- cls -> using cls token 
- reg -> using register output tokens
- patch -> using patch output tokens
- ...
The representations above are from a MAE. In theory this should be implemented differently

We want to support the following classification/regression heads:
- linear -> a linear classifier on top of the backbone's represeantation features
- attn -> uses a learned query token and scaled dot-product attention over the unpooled token sequence, then a linear classifier This should exist as a config, but shouldn't be implmented yet.

The code should be extensible -> it should be clear where and how to add additional variants of fine-tuning or classification heads or pooling mechanisms...

It's fine to have to write a small wrapper around an existing model to make it work with this finetuning/evals code.

Tasks:
- there should be a folder in evals where tasks can be defined
- task defines how data is downloaded, loaded, prepared etc.
- any custom things for that tasks are done there

Tasks can be:
- regression
- classification. This should exist as a config, but shouldn't be implmented yet.



We should be able to define yaml configs that can specify the evals: which model to use, which task, which representation, which classifier...


## Notes
Let's not reach for abstractions when that's not needed.
If this is complicated, we can focus on linear probing to start with. Let's analyze this when brainstorming.