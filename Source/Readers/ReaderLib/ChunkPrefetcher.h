//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <map>
#include <queue>
#include <condition_variable>
#include "DataDeserializer.h"
#include <set>
#include <thread>
#include <atomic>

namespace Microsoft { namespace MSR { namespace CNTK {
/*
template<typename Element>
class blocking_queue
{
    typedef std::unique_lock<std::mutex> guard_type;

    std::queue<Element> m_queue;
    

public:
    void push(const Element& elem)
    {
        {
            guard_type guard(m_lock);
            m_queue.push(elem);
        }

        m_notifier.notify_one();
    }

    bool empty() const
    {
        guard_type guard(m_lock);
        return m_queue.empty();
    }

    bool try_pop(Element& result)
    {
        guard_type guard(m_lock);
        if (m_queue.empty())
        {
            return false;
        }

        result = m_queue.front();
        m_queue.pop();
        return true;
    }

    void pop(Element& result)
    {
        guard_type guard(m_lock);
        while (m_queue.empty())
        {
            m_notifier.wait(guard);
        }

        result = m_queue.front();
        m_queue.pop();
    }

    void clear()
    {
        guard_type guard(m_lock);
        while (!m_queue.empty())
        {
            m_queue.pop();
        }
    }
};
*/

class ChunkPrefetcher
{
    typedef void (*DeleteThreadType)(std::thread* ptr);
    static void DeleteThread(std::thread* ptr)
    {
        ptr->join();
        delete ptr;
    }

public:
    ChunkPrefetcher(IDataDeserializerPtr deserializer) :
        m_deserializer(deserializer),
        m_prefetcher(nullptr, DeleteThread)
    {}

    void Start()
    {
        m_stopFlag = false;
        m_prefetcher = std::unique_ptr<std::thread, DeleteThreadType>(
            new std::thread(&ChunkPrefetcher::Process, this), DeleteThread);
    }

    void Stop()
    {
        m_stopFlag = true;
        m_notifier.notify_all();
        m_prefetcher = nullptr;
    }

    void Prefetch(const std::vector<ChunkIdType>& chunks)
    {
        std::vector<ChunkIdType> newChunks;
        for (auto chunkId : chunks)
        {
            if (m_chunkIds.find(chunkId) == m_chunkIds.end())
            {
                m_chunkIds.insert(chunkId);
                newChunks.push_back(chunkId);
            }
        }

        guard_type guard(m_idLock);
        m_toBePrefetched.insert(m_toBePrefetched.end(), newChunks.begin(), newChunks.end());
        m_notifier.notify_all();
    }

    ChunkPtr GetPrefetchedChunk(ChunkIdType chunkId)
    {
        if (m_chunkIds.find(chunkId) == m_chunkIds.end())
        {
            RuntimeError("Asked for not prefetched chunk.");
        }

        ChunkPtr result;
        {
            guard_type guard(m_dataLock);
            while (m_chunks.find(chunkId) == m_chunks.end())
            {
                m_notifier.wait(guard);
            }

            result = m_chunks[chunkId];
            m_chunks.erase(chunkId);
        }

        m_chunkIds.erase(chunkId);
        return result;
    }

    void Clear()
    {
        m_chunkIds.clear();

        {
            guard_type guard(m_idLock);
            m_toBePrefetched.clear();
        }

        {
            guard_type guard(m_dataLock);
            m_chunks.clear();
        }
    }

private:

    void Process()
    {
        std::vector<ChunkIdType> ids;
        while (!m_stopFlag)
        {
            {
                guard_type guard(m_idLock);
                while (m_toBePrefetched.empty() && !m_stopFlag)
                {
                    m_notifier.wait(guard);
                }

                ids.assign(m_toBePrefetched.begin(), m_toBePrefetched.end());
                m_toBePrefetched.clear();
            }

            // We have to lock the loop, so that 
            // if m_chunks gets cleared we are in consistent state.
            guard_type guard(m_dataLock);
            for (auto id : ids)
            {
                ChunkPtr chunk = m_deserializer->GetChunk(id);
                m_chunks[id] = chunk;
            }

            m_notifier.notify_all();
        }
    }

    std::set<ChunkIdType> m_chunkIds;
    std::vector<ChunkIdType> m_toBePrefetched;
    std::map<ChunkIdType, ChunkPtr> m_chunks;
    std::unique_ptr<std::thread, DeleteThreadType> m_prefetcher;
    std::atomic<bool> m_stopFlag;

    IDataDeserializerPtr m_deserializer;

    typedef std::unique_lock<std::mutex> guard_type;
    mutable std::mutex m_dataLock;
    mutable std::mutex m_idLock;
    std::condition_variable m_notifier;

    DISABLE_COPY_AND_MOVE(ChunkPrefetcher);
};

}}}
