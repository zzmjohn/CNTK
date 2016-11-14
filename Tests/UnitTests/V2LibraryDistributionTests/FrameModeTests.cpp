//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;
using namespace std::placeholders;

extern bool Is1bitSGDAvailable();

size_t g_outputFreqInMB = 20;
size_t g_minibatchSize = 25;

void TrainSimpleDistributedFeedForwardClassifer(const DeviceDescriptor& device, DistributedTrainerPtr distributedTrainer, size_t rank, std::vector<double>* trainCE=nullptr)
{
    UNUSED(rank);
    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    const size_t hiddenLayerDim = 50;
    const size_t numHiddenLayers = 2;

    const size_t numSamplesPerSweep = 10000;
    const size_t numSweepsToTrainWith = 2;
    const size_t numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / g_minibatchSize;

    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";
    auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } }, MinibatchSource::FullDataSweep, false);
    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>> inputMeansAndInvStdDevs = { { featureStreamInfo, { nullptr, nullptr } } };
    ComputeInputPerDimMeansAndInvStdDevs(minibatchSource, inputMeansAndInvStdDevs);

    auto nonLinearity = std::bind(Sigmoid, _1, L"Sigmoid");
    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto normalizedinput = PerDimMeanVarianceNormalize(input, inputMeansAndInvStdDevs[featureStreamInfo].first, inputMeansAndInvStdDevs[featureStreamInfo].second);
    auto classifierOutput = FullyConnectedDNNLayer(normalizedinput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedInput"));
    for (size_t i = 1; i < numHiddenLayers; ++i)
        classifierOutput = FullyConnectedDNNLayer(classifierOutput, hiddenLayerDim, device, nonLinearity, std::wstring(L"FullyConnectedHidden"));

    auto outputTimesParam = Parameter({ numOutputClasses, hiddenLayerDim }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputTimesParam");
    auto outputBiasParam = Parameter({ numOutputClasses }, DataType::Float, UniformInitializer(CNTK::DefaultParamInitScale, 1), device, L"outputBiasParam");
    classifierOutput = Plus(outputBiasParam, Times(outputTimesParam, classifierOutput), L"classifierOutput");

    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");;
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");

    auto learningRatePerSample = LearningRatePerSampleSchedule(0.02);
    minibatchSource = TextFormatMinibatchSource(L"SimpleDataTrain_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });
    Trainer trainer(classifierOutput, trainingLoss, prediction, { SGDLearner(classifierOutput->Parameters(), learningRatePerSample) }, distributedTrainer);
    if (trainCE) trainCE->clear();
    for (size_t i = 0; i < numMinibatchesToTrain; ++i)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(g_minibatchSize, device);
        trainer.TrainMinibatch({ { input, minibatchData[featureStreamInfo].m_data }, { labels, minibatchData[labelStreamInfo].m_data } }, device);
        PrintTrainingProgress(trainer, i, g_outputFreqInMB);

        if ((i % g_outputFreqInMB) == 0 && trainCE)
        {
            trainCE->push_back(trainer.PreviousMinibatchLossAverage());
        }
    }
}

void TestFrameMode()
{
    std::vector<double> CPUTrainCE;
    std::vector<double> GPUTrainCE;
    {
        auto communicator = MPICommunicator();
        auto distributedTrainer = CreateDataParallelDistributedTrainer(communicator, false);
        TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank, &CPUTrainCE);

        if (IsGPUAvailable())
        {
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank, &GPUTrainCE);

            for (int i = 0; i < CPUTrainCE.size(); i++)
            {
                FloatingPointCompare(CPUTrainCE[i], GPUTrainCE[i], "CPU/GPU training is not matching");
            }
        }
    }

    if (Is1bitSGDAvailable())
    {
        {
            size_t distributedAfterMB = 100;
            std::vector<double> TrainCE;
            size_t distributedAfterSampleCount = distributedAfterMB * g_minibatchSize;

            auto communicator = QuantizedMPICommunicator(true, true, 1);
            auto distributedTrainer = CreateQuantizedDataParallelDistributedTrainer(communicator, false, distributedAfterSampleCount);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank, &TrainCE);

            for (int i = 0; i < distributedAfterMB / g_outputFreqInMB; i++)
            {
                FloatingPointCompare(TrainCE[i], CPUTrainCE[i], "Warm start CE deviated from non-distributed");
            }

            if (IsGPUAvailable())
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
        }

        {
            auto communicator = MPICommunicator();
            auto distributedTrainer = CreateBlockMomentumDistributedTrainer(communicator, 1024);
            TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::CPUDevice(), distributedTrainer, communicator->CurrentWorker().m_globalRank);

            if (IsGPUAvailable())
                TrainSimpleDistributedFeedForwardClassifer(DeviceDescriptor::GPUDevice(0), distributedTrainer, communicator->CurrentWorker().m_globalRank);
        }
    }
}