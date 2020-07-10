:github_url: https://github.com/beta-team/beta-recsys


.. image:: _static/img/Logo.svg

Beta-RecSys Documentation
===============================

Beta-RecSys an open source project for Building, Evaluating and Tuning Automated Recommender Systems.
Beta-RecSys aims to provide a practical data toolkit for building end-to-end recommendation systems in a standardized way.
It provided means for dataset preparation and splitting using common strategies, a generalized model engine for implementing recommender models using Pytorch with a lot of models available out-of-the-box,
as well as a unified training, validation, tuning and testing pipeline. Furthermore, Beta-RecSys is designed to be both modular and extensible, enabling new models to be quickly added to the framework.
It is deployable in a wide range of environments via pre-built docker containers and supports distributed parameter tuning using Ray.

================================

.. image:: https://codecov.io/gh/leungyukshing/beta-recsys/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/leungyukshing/beta-recsys

.. image:: https://github.com/beta-team/beta-recsys/workflows/CI/badge.svg?branch=develop
  :target: https://github.com/beta-team/beta-recsys/actions

.. toctree::
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction
   notes/framework
   notes/datasets
   notes/dataloaders
   notes/models
   notes/evaluation
   notes/tuning

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   modules/core
   modules/data
   modules/datasets
   modules/models
   modules/utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
