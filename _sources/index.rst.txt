.. BrainAge documentation master file, created by
   sphinx-quickstart on Mon May 13 19:01:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BrainAge's documentation!
====================================
BrainAge is a python library whose aim is to predict age of patients analizing their
brain features extracted from MRI (magnetic resonance imaging).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
------------

The library contains programs performing statistical analysis on a preprocessed dataset of brain features. Input 
data is a *.csv* file whose columns are features of patients and rows are different patients. 
Statistical programs provided are a feed-foward neural network and some regression models; there are also some minor programs 
necessary for the correct functioning of the library. 

Contents
--------

.. toctree::

   Usage
   API Reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`