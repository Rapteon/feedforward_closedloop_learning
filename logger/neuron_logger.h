#ifndef NEURONLOGGER_H
#define NEURONLOGGER_H

#include "fcl.h"
#include "logger.h"

class NeuronLogger:public Logger {
    public:
    NeuronLogger(FCLNeuron* neuron, const char* fileName);
};

#endif