#ifndef __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_
#define __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_

#include "fcl/globals.h"
#include "fcl/layer.h"
#include "fcl/neuron.h"
#include "fcl/bandpass.h"
#include "fcl.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/**
 * Derived classes of the FeedforwardClosedloopLearning class
 * for special functionality
 **/


class FeedforwardClosedloopLearningWithFilterbank : public FeedforwardClosedloopLearning {
	/**
	 * FeedforwardClosedloopLearning with Filterbank at each input
	 **/
public:
	/** Constructor: FCL with a filter bank at the input
	 * Every input feeds internally into a has a filter bank of num_filtersInput 
	 * filters. This allows for a temporal distribution of the inputs.
	 * \param num_of_inputs Number of inputs in the input layer
	 * \param num_of_neurons_per_layer_array Number of neurons in each layer
	 * \param num_filtersInput Number of filters at the input layer, 0 = no filterbank
	 * \param num_filters Number of filters in the hiddel layers (usually zero)
	 * \param _minT Minimum/first temporal duration of the 1st filter
	 * \param _maxT Maximum/last temporal duration of the last filter
	 **/
	FeedforwardClosedloopLearningWithFilterbank(
			const int num_of_inputs,
			const std::vector<int> &num_of_neurons_per_layer,
			const int num_filtersInput,
			const double minT,
			const double maxT);

	/**
	 * Destructor
	 **/
	~FeedforwardClosedloopLearningWithFilterbank();

	/** Performs the simulation step
         * \param input Array with the input values
         * \param error Array of the error signals
         **/
	void doStep(const std::vector<double> &input, const std::vector<double> &error);

	double getFilterOutput(int inputIdx, int filterIdx) {
		const int idx = inputIdx * nFiltersPerInput + filterIdx;
		assert((idx >= 0) || (idx < (nFiltersPerInput * nInputs)));
		return filterbankOutputs[idx];
	}

	int getNFiltersPerInput() {
		return nFiltersPerInput;
	}

	/**
	 * Logs state of the network into the file
	 */
	void logState(const std::vector<double> &input) {
		for (unsigned int i = 0; i < input.size()-1; ++i) {
			fprintf(inputlog, "%e\t", input[i]);
			fflush(inputlog);
		}
		fprintf(inputlog, "%e\n", input[input.size()-1]);
		fflush(inputlog);
	}

	void logState(FCLLayer* layer, FILE* log_file) {
		// Printing each neuron's inputs
		for (int i {0}; i < layer->getNneurons(); ++i) {
			auto neuron = layer->getNeuron(i);
			for (int j {0}; j < neuron->getNinputs(); ++j) {
				fprintf(log_file, "%e\t", neuron->getInput(j));
				fflush(log_file);
			}
		}
		// Printing errors of each neuron
		for (int i {0}; i < layer->getNneurons(); ++i) {
			auto neuron = layer->getNeuron(i);
			fprintf(log_file, "%e\t", neuron->getError());
			fflush(log_file);
		}

		// Printing weights between each input of each neuron
		for (int i {0}; i < layer->getNneurons(); ++i) {
			auto neuron = layer->getNeuron(i);
			for (int j {0}; j < neuron->getNinputs(); ++j) {
				fprintf(log_file, "%e\t", neuron->getWeight(j));
				fflush(log_file);
			}
		}
		// Printing bias weights of each neuron
		for (int i {0}; i < layer->getNneurons(); ++i) {
			auto neuron = layer->getNeuron(i);
			fprintf(log_file, "%e\t", neuron->getBiasWeight());
			fflush(log_file);
		}
		
		// Printing outputs of each neuron
		for (int i {0}; i < layer->getNneurons(); ++i) {
			auto neuron = layer->getNeuron(i);
			fprintf(log_file, "%e\t", neuron->getOutput());
			fflush(log_file);
		}

		fprintf(log_file, "\n");
	}

	void logState() {
		auto layer1 = getLayer(0);
		fprintf(layer_props, "%d\t%d\n", layer1->getNneurons(), layer1->getNinputs());
		auto layer2 = getLayer(1);
		fprintf(layer_props, "%d\t%d\n", layer2->getNneurons(), layer2->getNinputs());
		auto layer3 = getLayer(2);
		fprintf(layer_props, "%d\t%d\n", layer3->getNneurons(), layer3->getNinputs());
		fflush(layer_props);
	}
private:
	const double dampingCoeff = 0.51;
	FCLBandpass ***bandpass = 0;
	std::vector<double> errors;
	std::vector<double> filterbankOutputs;
	int nFiltersPerInput = 0;
	int nInputs = 0;
	FILE* inputlog;
	FILE* layer_1_log;
	FILE* layer_2_log;
	FILE* layer_3_log;
	FILE* layer_props;
};

#endif
