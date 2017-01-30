# Text to phoneme project
Deep Learning for Speech and Language

To design, implement and train a neural network to predict the phonemes of a word given it as a text.

A presentation of the project can be found in (may require additional permission):
https://docs.google.com/presentation/d/1gp6NUnFIWxY4sGYGnY6zQMN-T3wniQhM5MK3-oAuwWI/edit#slide=id.g1bb33be1e7_0_20

- pho_rnn.py is the main programme (that trains and tests the network)
- compare_results.py and compare_results_several.py are used to compare the results of different simulations in a plot
- tasas.c calculates the Goals, Substitutions, Erasures and Insertions done by the network given the predicted and the correct phonemes
- wcmudict... are the databases for training and testing the network. Aligned means that the relation letter/phoneme is one-to-one
- Simulation_results/ contains the simulation results