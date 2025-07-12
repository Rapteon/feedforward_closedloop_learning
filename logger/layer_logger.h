#ifndef LAYER_LOGGER_H
#define LAYER_LOGGER_H

#include "fcl.h"
#include "logger.h"

class LayerLogger : public Logger
{
public:
	LayerLogger(FCLLayer *layer, const char *fileName);
	void log();

private:
	FCLLayer *layer;
};

#endif