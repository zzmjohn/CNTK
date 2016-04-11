//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS

#include "SequenceRandomizer.h"
#include <algorithm>
#include <utility>
#include <deque>

#include "DataReader.h"
#include <random>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

    // NOTE: This is an old code, used for legacy randomization to make sure we preserve the same behavior for the tests.
    // TODO: Deprecate when the new randomizer is in place.
    static inline size_t rand(const size_t begin, const size_t end)
    {
        // still only covers 32-bit range
        const size_t randomNumber = ::rand() * RAND_MAX + ::rand();
        return begin + randomNumber % (end - begin);
    }

    SequenceRandomizer::SequenceRandomizer(
        IDataDeserializerPtr deserializer,
        ChunkRandomizerPtr chunkRandomizer)
        : m_randomizedChunks(chunkRandomizer->GetRandomizedChunks()),
        m_h(0),
        m_k(0),
        m_nextChunkNotYetRandomized(0),
        m_currentSequencePosition(0),
        m_currentChunkPosition(0),
        m_deserializer(deserializer)
    {
        size_t max = 0;
        for (const auto& c : m_randomizedChunks)
        {
            if (max < c.m_original->m_numberOfSequences)
            {
                max = c.m_original->m_numberOfSequences;
            }
        }

        m_bufferOriginalSequences.reserve(max);
    }

    // Gets next randomized sequence descriptions not exceeding the count.
    std::vector<RandomizedSequenceDescription> SequenceRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
    {
        int samples = (int)sampleCount;

        std::vector<RandomizedSequenceDescription> result;
        result.reserve(sampleCount);

        size_t sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
        RandomizedSequenceDescription* sequence = &m_sequenceWindow[m_currentChunkPosition - m_h][sequenceOffsetInsideChunk];

        result.push_back(*sequence);
        samples -= (int)sequence->m_numberOfSamples;
        m_currentSequencePosition++;

        if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->m_numberOfSequences)
        {
            // Moving to the next chunk.
            MoveChunkCursor();
        }

        while (samples > 0 && m_currentChunkPosition < m_randomizedChunks.size())
        {
            sequenceOffsetInsideChunk = m_currentSequencePosition - m_randomizedChunks[m_currentChunkPosition].m_sequencePositionStart;
            sequence = &m_sequenceWindow[m_currentChunkPosition - m_h][sequenceOffsetInsideChunk];
            if (samples - sequence->m_numberOfSamples >= 0)
            {
                result.push_back(*sequence);
                m_currentSequencePosition++;
                samples -= (int)sequence->m_numberOfSamples;

                if (sequenceOffsetInsideChunk + 1 >= m_randomizedChunks[m_currentChunkPosition].m_original->m_numberOfSequences)
                {
                    // Moving to the next chunk.
                    MoveChunkCursor();
                }
            }
        }

        return result;
    }

    void SequenceRandomizer::MoveChunkCursor()
    {
        m_currentChunkPosition++;
        RandomizeNextChunkIfNeeded();
    }

    void SequenceRandomizer::RandomizeNextChunkIfNeeded()
    {
        if (m_currentChunkPosition < m_i)
        {
            assert(m_currentChunkPosition >= m_h);
            return;
        }
        assert(m_i == m_currentChunkPosition);

        if (m_i == m_randomizedChunks.size())
        {
            return;
        }

        // Chunk not yet randomized.
        // of the sample position we have to randomized (current + sampleCount).
        // We will randomize up to this chunk as the final position of windows end is guaranteed to have been determined
        // when all sequences up to that chunk have been randomized
        size_t endChunkIdxToRandomize = m_randomizedChunks[m_i].m_randomizationWindow.m_end;
        while (endChunkIdxToRandomize < m_randomizedChunks.size() &&
               m_randomizedChunks[endChunkIdxToRandomize].m_randomizationWindow.m_begin <= m_i)
        {
            endChunkIdxToRandomize++;  // new J
        }

        // TODO: we should drop chunks, but firstly make sure that they are not used any more.
        // TODO: That means the sequence description that we have got from the previous call can still be in the BlockRandomizer,
        // TODO: so we need to make sure that the clean up code below is used only when the chunk is not required anymore.
        // size_t candiateToUnload = m_h;
        // while (candiateToUnload < m_randomizedChunks[m_i].m_randomizationWindow.m_begin)
        // {
            // Can unload
            // if (m_randomizedChunks[candiateToUnload].m_randomizationWindow.m_end <= m_i)
            //{
            //    m_sequenceWindow.pop_front();
            //    m_chunkWindow.pop_front();
            //    m_randomizedChunkInfo.pop_front();
            //    m_h++;
            //}
        // }

        // Determine the end chunk that we need to load into memory.
        size_t endChunkIdx = m_randomizedChunks[endChunkIdxToRandomize - 1].m_randomizationWindow.m_end; // new K

        // Lets page in everything from m_currentRangeEndChunkIndex to endChunkIdx
        for (size_t i = m_k; i < endChunkIdx; ++i)
        {
            AddRandomizedSequencesForChunk(i);
        }

        size_t firstSequencePositionToRandomize = m_randomizedChunks[m_j].m_sequencePositionStart;
        size_t endSequencePosToRandomize = m_randomizedChunks[endChunkIdxToRandomize - 1].SequenceEndPosition();
        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // Get valid randomization range, expressed in chunks
            // TODO: This can be done more efficiently, we know the range of chunks already.
            const size_t currentChunkIdx = GetChunkIndexForSequencePosition(t);

            size_t chunkWindowBegin = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_begin;
            size_t chunkWindowEnd = m_randomizedChunks[currentChunkIdx].m_randomizationWindow.m_end;

            // Get valid randomization range, expressed in sequence positions.
            size_t posBegin = m_randomizedChunks[chunkWindowBegin].m_sequencePositionStart;
            size_t posEnd = m_randomizedChunks[chunkWindowEnd - 1].SequenceEndPosition();

            for (;;)
            {
                // Pick a sequence position from [posBegin, posEnd)
                const size_t j = rand(posBegin, posEnd);

                // Try again if the sequence currently at j cannot be placed at position i.
                if (!IsValidForPosition(t, GetRandomizedSequenceDescriptionBySequenceId(j)))
                    continue;

                // Try again if the sequence currently at i cannot be placed at position j.
                if (!IsValidForPosition(j, GetRandomizedSequenceDescriptionBySequenceId(t)))
                    continue;

                // Swap and break out.
                std::swap(GetRandomizedSequenceDescriptionBySequenceId(t), GetRandomizedSequenceDescriptionBySequenceId(j)); // TODO old swap was perhaps more efficient
                break;
            }
        }

        // Verify that we got it right
        for (size_t t = firstSequencePositionToRandomize; t < endSequencePosToRandomize; ++t)
        {
            // TODO assert only
            if (!IsValidForPosition(t, GetRandomizedSequenceDescriptionBySequenceId(t)))
            {
                LogicError("SequenceRandomizer::RandomizeNextSequenceDescriptions: randomization logic mangled!");
            }
        }

        // Let's recalculate number of samples in the randomized chunks for efficient indexing in seek.
        size_t sampleCount = 0;
        size_t randomizedChunk = m_i - m_h;
        for (size_t index = 0; index < m_sequenceWindow[randomizedChunk].size(); index++)
        {
            sampleCount += m_sequenceWindow[randomizedChunk][index].m_numberOfSamples;
        }

        // Safe the sample information.
        ChunkInfo info;
        info.numberOfSamples = sampleCount;
        info.start = m_randomizedChunkInfo.empty() ? 0 : m_randomizedChunkInfo.back().start + m_randomizedChunkInfo.back().numberOfSamples;
        m_randomizedChunkInfo.push_back(info);

        // Update the cursors.
        m_i++;
        m_j = endChunkIdxToRandomize;
        m_k = endChunkIdx;
    }

    // Resets the current sweep according to the randomization seed provided.
    void SequenceRandomizer::Reset(size_t randSeed)
    {
        srand((unsigned int)randSeed);

        m_sequenceWindow.clear();
        m_chunkWindow.clear();
        m_randomizedChunkInfo.clear();
        m_h = m_i = m_j = m_k = 0;
        m_nextChunkNotYetRandomized = 0;
        m_currentSequencePosition = 0;
        m_currentChunkPosition = 0;

        // Prepare the chunk for reading
        RandomizeNextChunkIfNeeded();
    }

    // Sets current sequence position to the sample offset.
    // If offset is in the middle of the sequence, the next sequence is picked up.
    size_t SequenceRandomizer::Seek(size_t offset, size_t sweep)
    {
        size_t hs = m_randomizedChunkInfo.empty() ? 0 : m_randomizedChunkInfo.front().start;
        size_t is = m_randomizedChunkInfo.empty() ? 0 : m_randomizedChunkInfo.back().start + m_randomizedChunkInfo.back().numberOfSamples;
        if(offset < hs)
        {
            Reset(sweep + 1);
        }
        else if (offset < is)
        {
            size_t index;
            for (index = 0; index <= m_randomizedChunkInfo.size() - 1; index++)
            {
                if (m_randomizedChunkInfo[index].start >= offset && offset < (m_randomizedChunkInfo[index].start + m_randomizedChunkInfo[index].numberOfSamples))
                {
                    break;
                }
            }
            m_currentCursor = m_randomizedChunkInfo[index].start;
        }

        // advance
        // offset - is
        while (m_currentCursor < offset)
        {
            GetNextSequenceDescriptions(1);
        }

        return m_currentCursor;
    }

    // Checks if the randomized sequence is valid for a target position using its chunk randomization window.
    bool SequenceRandomizer::IsValidForPosition(size_t targetPosition, const RandomizedSequenceDescription& seqDesc) const
    {
        const auto& chunk = m_randomizedChunks[GetChunkIndexForSequencePosition(targetPosition)];
        return chunk.m_randomizationWindow.m_begin <= seqDesc.m_chunk->m_chunkId && seqDesc.m_chunk->m_chunkId < chunk.m_randomizationWindow.m_end;
    }

    // Gets chunk index using a sequence position in the sweep.
    // TODO: upper bound should be used instead.
    size_t SequenceRandomizer::GetChunkIndexForSequencePosition(size_t sequencePosition) const
    {
        struct PositionConverter
        {
            size_t m_position;
            PositionConverter(const RandomizedChunk & chunk) : m_position(chunk.m_sequencePositionStart) {};
            PositionConverter(size_t sequencePosition) : m_position(sequencePosition) {};
        };

        auto result = std::lower_bound(m_randomizedChunks.begin(), m_randomizedChunks.end(), sequencePosition,
            [](const PositionConverter& a, const PositionConverter& b)
        {
            return a.m_position <= b.m_position;
        });

        return result - 1 - m_randomizedChunks.begin();
    }

    // Add randomizes sequences for the chunk with a given index.
    void SequenceRandomizer::AddRandomizedSequencesForChunk(size_t chunkIdx)
    {
        assert(chunkIdx == m_k);

        const RandomizedChunk& chunk = m_randomizedChunks[chunkIdx];
        std::vector<RandomizedSequenceDescription> chunkSequences;

        m_bufferOriginalSequences.clear();
        m_deserializer->GetSequencesForChunk(chunk.m_original->m_id, m_bufferOriginalSequences);
        chunkSequences.reserve(m_bufferOriginalSequences.size());
        for (size_t k = 0; k < m_bufferOriginalSequences.size(); k++)
        {
            RandomizedSequenceDescription s;
            s.m_id = m_bufferOriginalSequences[k].m_id;
            s.m_numberOfSamples = m_bufferOriginalSequences[k].m_numberOfSamples;
            s.m_chunk = &chunk;
            chunkSequences.push_back(s);
        }

        m_sequenceWindow.push_back(std::move(chunkSequences));
        m_chunkWindow.push_back(chunk);
        m_k++;
    }

    // Gets randomized sequence by the sequence id.
    RandomizedSequenceDescription& SequenceRandomizer::GetRandomizedSequenceDescriptionBySequenceId(size_t sequenceId)
    {
        size_t globalChunkIdx = GetChunkIndexForSequencePosition(sequenceId);
        size_t sequenceOffsetInsideChunk = sequenceId - m_randomizedChunks[globalChunkIdx].m_sequencePositionStart;
        return m_sequenceWindow[globalChunkIdx - m_h][sequenceOffsetInsideChunk];
    }
}}}
