/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob
    Copyright (c) 2017 by ETH Zurich, Thomas Mueller.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <optional>
#include <map>
#include <queue>
#include <algorithm>

MTS_NAMESPACE_BEGIN

double statsSdtreeBuild = 0.0;
double statsSdtreeReset = 0.0;
double statsAMISSdtreeExtra = 0.0;
double statsAMISTimeArchive = 0.0;
double statsAMISTimeSplat = 0.0;
double statsPhaseTimeRendering = 0.0;
double statsPhaseTimeRenderPass = 0.0;
double statsPhaseTimeTotal = 0.0;
double statsPhaseTimeSampleMat = 0.0;
double statsPhaseTimeCommit = 0.0;
double statsPhaseTimeRenderBlockSum = 0.0;
double statsPhaseTimeRenderPostproc = 0.0;
double statsPhaseTimeRenderRecording1 = 0.0;
double statsPhaseTimeRenderRecording = 0.0;
int64_t statsSuperfuseDFSCall = 0;
int64_t statsSuperfusePushdownCall = 0;
int64_t statsResetBFSCall = 0;
int64_t statsCommitCall = 0;
int64_t statsCommitRequestTotal = 0;
int64_t statsImageSamples = 0;
int64_t statsImageSamplesNonzero = 0;
int64_t statsImageSamplesAMIS = 0;
int64_t statsRecordedVertices = 0;

int g_sampleCount = 0;
int g_passesThisIteration = 0;
float g_selectiveActivateThreshold = 0;
float g_tempParam = 0; // currently not use. You can use it for any experimental purpose.
ref<Sensor> g_sensor;
Point3f g_first_vertex;

void printMystats()
{
    printf("Guided path tracer: Sdtree Build = %.6f\n", statsSdtreeBuild);
    printf("Guided path tracer: Sdtree Reset = %.6f\n", statsSdtreeReset);
    printf("Guided path tracer: AMIS Sdtree Extra = %.6f\n", statsAMISSdtreeExtra);
    printf("Guided path tracer: Sdtree All = %.6f\n", statsSdtreeBuild + statsSdtreeReset + statsAMISSdtreeExtra);
    puts("");
    printf("Guided path tracer: AMIS Time Archive = %.6f\n", statsAMISTimeArchive);
    printf("Guided path tracer: AMIS Time Splat = %.6f\n", statsAMISTimeSplat);
    puts("");
    printf("Guided path tracer: Phase Time Rendering = %.6f\n", statsPhaseTimeRendering);
    printf("Guided path tracer: Phase Time RenderPass = %.6f\n", statsPhaseTimeRenderPass);
    printf("Guided path tracer: Phase Time   SampleMat = %.6f\n", statsPhaseTimeSampleMat);
    printf("Guided path tracer: Phase Time   Commit = %.6f\n", statsPhaseTimeCommit);
    printf("Guided path tracer: Phase Time   statsPhaseTimeRenderBlockSum = %.6f\n", statsPhaseTimeRenderBlockSum);
    printf("Guided path tracer: Phase Time   statsPhaseTimeRenderRecording1 = %.6f\n", statsPhaseTimeRenderRecording1);
    printf("Guided path tracer: Phase Time   statsPhaseTimeRenderRecording = %.6f\n", statsPhaseTimeRenderRecording);
    printf("Guided path tracer: Phase Time RenderPostproc= %.6f\n", statsPhaseTimeRenderPostproc);
    puts("");
    printf("Guided path tracer: statsSuperfuseDFSCall = %lld\n", statsSuperfuseDFSCall);
    printf("Guided path tracer: statsSuperfusePushdownCall = %lld\n", statsSuperfusePushdownCall);
    printf("Guided path tracer: statsResetBFSCall = %lld\n", statsResetBFSCall);
    puts("");
    printf("Guided path tracer: statsCommitCall = %lld\n", statsCommitCall);
    printf("Guided path tracer: statsCommitRequestTotal = %lld\n", statsCommitRequestTotal);
    printf("Guided path tracer:   accept rate = %.4f \%\n", statsCommitCall * 100.0f / statsCommitRequestTotal);
    puts("");
    printf("Guided path tracer: statsImageSamples = %lld\n", statsImageSamples);
    printf("Guided path tracer: statsImageSamplesNonzero = %lld\n", statsImageSamplesNonzero);
    printf("Guided path tracer: statsImageSamplesAMIS = %lld\n", statsImageSamplesAMIS);
    printf("Guided path tracer:   nonzero rate = %.4f \%\n", statsImageSamplesNonzero * 100.0f / statsImageSamples);
    printf("Guided path tracer:   amis rate = %.4f \%\n", statsImageSamplesAMIS * 100.0f / statsImageSamples);
    printf("Guided path tracer:   amis rate nonzero = %.4f \%\n", statsImageSamplesAMIS * 100.0f / statsImageSamplesNonzero);
    printf("Guided path tracer: statsRecordedVertices = %lld\n", statsRecordedVertices);
    printf("Guided path tracer: statsRecordedVertices mem = %.3f\n", statsRecordedVertices * 16.0 / 1048576);
}

class HDTimer
{
public:
    using Unit = std::chrono::nanoseconds;

    HDTimer()
    {
        start = std::chrono::system_clock::now();
    }

    double value() const
    {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        return (double)duration.count() * 1e-9;
    }

    double reset()
    {
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<Unit>(now - start);
        start = now;
        return (double)duration.count() * 1e-9;
    }

    static std::chrono::system_clock::time_point staticValue()  
    {
        return std::chrono::system_clock::now();
    }

    static double staticDelta(std::chrono::system_clock::time_point t)  
    {
        return std::chrono::duration_cast<Unit>(std::chrono::system_clock::now() - t).count() * 1e-9;
    }

private:
    std::chrono::system_clock::time_point start;
};

float computeElapsedSeconds(std::chrono::steady_clock::time_point start)
{
    auto current = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
    return (float)ms.count() / 1000;
}

struct RawImageSample
{
    std::vector<Point4f> path; 
    Point2f last_dir; // 2*max_quadtree_depth bits at most actually
    Spectrum value;
    int iter; // uint8_t actually, neglectable
    float original_radiance; // not really needed

    bool operator<(const RawImageSample &rhs) const
    {
        return original_radiance > rhs.original_radiance;
    }

    std::string toString() const
    {
    }
};

class BlobWriter
{
public:
    BlobWriter(const std::string &filename)
        : f(filename, std::ios::out | std::ios::binary)
    {
    }

    template <typename Type>
    typename std::enable_if<std::is_standard_layout<Type>::value, BlobWriter &>::type
    operator<<(Type Element)
    {
        Write(&Element, 1);
        return *this;
    }

    // CAUTION: This function may break down on big-endian architectures.
    //          The ordering of bytes has to be reverted then.
    template <typename T>
    void Write(T *Src, size_t Size)
    {
        f.write(reinterpret_cast<const char *>(Src), Size * sizeof(T));
    }

private:
    std::ofstream f;
};

static void addToAtomicFloat(std::atomic<float> &var, float val)
{
    auto current = var.load();
    while (!var.compare_exchange_weak(current, current + val))
        ;
}

inline float logistic(float x)
{
    return 1 / (1 + std::exp(-x));
}

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer
{
public:
    AdamOptimizer(float learningRate, int batchSize = 1, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.999f)
    {
        m_hparams = {learningRate, batchSize, epsilon, beta1, beta2};
    }

    AdamOptimizer &operator=(const AdamOptimizer &arg)
    {
        m_state = arg.m_state;
        m_hparams = arg.m_hparams;
        return *this;
    }

    AdamOptimizer(const AdamOptimizer &arg)
    {
        *this = arg;
    }

    void append(float gradient, float statisticalWeight)
    {
        m_state.batchGradient += gradient * statisticalWeight;
        m_state.batchAccumulation += statisticalWeight;

        if (m_state.batchAccumulation > m_hparams.batchSize)
        {
            step(m_state.batchGradient / m_state.batchAccumulation);

            m_state.batchGradient = 0;
            m_state.batchAccumulation = 0;
        }
    }

    void step(float gradient)
    {
        ++m_state.iter;

        float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - std::pow(m_hparams.beta2, m_state.iter)) / (1 - std::pow(m_hparams.beta1, m_state.iter));
        m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
        m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
        m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

        // Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
        // since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
        // in *extremely* small and large results that are pretty much never necessary in practice.
        m_state.variable = std::min(std::max(m_state.variable, -20.0f), 20.0f);
    }

    float variable() const
    {
        return m_state.variable;
    }

private:
    struct State
    {
        int iter = 0;
        float firstMoment = 0;
        float secondMoment = 0;
        float variable = 0;

        float batchAccumulation = 0;
        float batchGradient = 0;
    } m_state;

    struct Hyperparameters
    {
        float learningRate;
        int batchSize;
        float epsilon;
        float beta1;
        float beta2;
    } m_hparams;
};

enum class ESampleCombination
{
    EAMIS,
};

enum class EBsdfSamplingFractionLoss
{
    ENone,
    EKL,
    EVariance,
};

enum class ESpatialFilter
{
    ENearest,
    EStochasticBox,
    EBox,
};

enum class EDirectionalFilter
{
    ENearest,
    EBox,
};

enum class ESampleAllocSeq
{
    EDouble,
    EUniform,
    EHalfdouble,
};

class QuadTreeNode
{
public:
    QuadTreeNode()
    {
        m_children = {};
        for (size_t i = 0; i < m_sum.size(); ++i)
        {
            m_sum[i].store(0, std::memory_order_relaxed);
        }
    }

    void setSum(int index, float val)
    {
        m_sum[index].store(val, std::memory_order_relaxed);
    }

    float sum(int index) const
    {
        return m_sum[index].load(std::memory_order_relaxed);
    }

    void copyFrom(const QuadTreeNode &arg)
    {
        for (int i = 0; i < 4; ++i)
        {
            setSum(i, arg.sum(i));
            m_children[i] = arg.m_children[i];
        }
    }

    QuadTreeNode(const QuadTreeNode &arg)
    {
        copyFrom(arg);
    }

    QuadTreeNode &operator=(const QuadTreeNode &arg)
    {
        copyFrom(arg);
        return *this;
    }

    void setChild(int idx, uint16_t val)
    {
        m_children[idx] = val;
    }

    uint16_t child(int idx) const
    {
        return m_children[idx];
    }

    void setSum(float val)
    {
        for (int i = 0; i < 4; ++i)
        {
            setSum(i, val);
        }
    }

    int childIndex(Point2 &p) const
    {
        int res = 0;
        for (int i = 0; i < Point2::dim; ++i)
        {
            if (p[i] < 0.5f)
            {
                p[i] *= 2;
            }
            else
            {
                p[i] = (p[i] - 0.5f) * 2;
                res |= 1 << i;
            }
        }

        return res;
    }

    // Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
    // To obtain radiance, the sum density (result of this function) must be divided
    // by the total statistical weight of the estimates that were summed up.
    float eval(Point2 &p, const std::vector<QuadTreeNode> &nodes) const
    {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index))
        {
            return 4 * sum(index);
        }
        else
        {
            return 4 * nodes[child(index)].eval(p, nodes);
        }
    }

    float pdf(Point2 &p, const std::vector<QuadTreeNode> &nodes) const
    {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (!(sum(index) > 0))
        {
            return 0;
        }

        const float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
        if (isLeaf(index))
        {
            return factor;
        }
        else
        {
            return factor * nodes[child(index)].pdf(p, nodes);
        }
    }

    int depthAt(Point2 &p, const std::vector<QuadTreeNode> &nodes) const
    {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        const int index = childIndex(p);
        if (isLeaf(index))
        {
            return 1;
        }
        else
        {
            return 1 + nodes[child(index)].depthAt(p, nodes);
        }
    }

    Point2 sample(Sampler *sampler, const std::vector<QuadTreeNode> &nodes) const
    {
        int index = 0;

        float topLeft = sum(0);
        float topRight = sum(1);
        float partial = topLeft + sum(2);
        float total = partial + topRight + sum(3);

        // Should only happen when there are numerical instabilities.
        if (!(total > 0.0f))
        {
            return sampler->next2D();
        }

        float boundary = partial / total;
        Point2 origin = Point2{0.0f, 0.0f};

        float sample = sampler->next1D();

        if (sample < boundary)
        {
            SAssert(partial > 0);
            sample /= boundary;
            boundary = topLeft / partial;
        }
        else
        {
            partial = total - partial;
            SAssert(partial > 0);
            origin.x = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            boundary = topRight / partial;
            index |= 1 << 0;
        }

        if (sample < boundary)
        {
            sample /= boundary;
        }
        else
        {
            origin.y = 0.5f;
            sample = (sample - boundary) / (1.0f - boundary);
            index |= 1 << 1;
        }

        if (isLeaf(index))
        {
            return origin + 0.5f * sampler->next2D();
        }
        else
        {
            return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
        }
    }

    void record(Point2 &p, float irradiance, std::vector<QuadTreeNode> &nodes)
    {
        SAssert(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
        int index = childIndex(p);

        if (isLeaf(index))
        {
            addToAtomicFloat(m_sum[index], irradiance);
        }
        else
        {
            nodes[child(index)].record(p, irradiance, nodes);
        }
    }

    float computeOverlappingArea(const Point2 &min1, const Point2 &max1, const Point2 &min2, const Point2 &max2)
    {
        float lengths[2];
        for (int i = 0; i < 2; ++i)
        {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1];
    }

    void record(const Point2 &origin, float size, Point2 nodeOrigin, float nodeSize, float value, std::vector<QuadTreeNode> &nodes)
    {
        float childSize = nodeSize / 2;
        for (int i = 0; i < 4; ++i)
        {
            Point2 childOrigin = nodeOrigin;
            if (i & 1)
            {
                childOrigin[0] += childSize;
            }
            if (i & 2)
            {
                childOrigin[1] += childSize;
            }

            float w = computeOverlappingArea(origin, origin + Point2(size), childOrigin, childOrigin + Point2(childSize));
            if (w > 0.0f)
            {
                if (isLeaf(i))
                {
                    addToAtomicFloat(m_sum[i], value * w);
                }
                else
                {
                    nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
                }
            }
        }
    }

    bool isLeaf(int index) const
    {
        return child(index) == 0;
    }

    // Ensure that each quadtree node's sum of irradiance estimates
    // equals that of all its children.
    void build(std::vector<QuadTreeNode> &nodes)
    {
        for (int i = 0; i < 4; ++i)
        {
            // During sampling, all irradiance estimates are accumulated in
            // the leaves, so the leaves are built by definition.
            if (isLeaf(i))
            {
                continue;
            }

            QuadTreeNode &c = nodes[child(i)];

            // Recursively build each child such that their sum becomes valid...
            c.build(nodes);

            // ...then sum up the children's sums.
            float sum = 0;
            for (int j = 0; j < 4; ++j)
            {
                sum += c.sum(j);
            }
            setSum(i, sum);
        }
    }

private:
    std::array<std::atomic<float>, 4> m_sum;
    std::array<uint16_t, 4> m_children;
};

class DTree
{
public:
    DTree()
    {
        m_atomic.sum.store(0, std::memory_order_relaxed);
        m_maxDepth = 0;
        m_nodes.emplace_back();
        m_nodes.front().setSum(0.0f);
    }

    const QuadTreeNode &node(size_t i) const
    {
        return m_nodes[i];
    }

    float mean() const
    {
        if (m_atomic.statisticalWeight == 0)
        {
            return 0;
        }
        const float factor = 1 / (M_PI * 4 * m_atomic.statisticalWeight);
        return factor * m_atomic.sum;
    }

    void recordIrradiance(Point2 p, float irradiance, float statisticalWeight, EDirectionalFilter directionalFilter)
    {
        if (std::isfinite(statisticalWeight) && statisticalWeight > 0)
        {
            addToAtomicFloat(m_atomic.statisticalWeight, statisticalWeight);

            if (std::isfinite(irradiance) && irradiance > 0)
            {
                if (directionalFilter == EDirectionalFilter::ENearest)
                {
                    m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
                }
                else
                {
                    int depth = depthAt(p);
                    float size = std::pow(0.5f, depth);

                    Point2 origin = p;
                    origin.x -= size / 2;
                    origin.y -= size / 2;
                    m_nodes[0].record(origin, size, Point2(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
                }
            }
        }
    }

    float pdf(Point2 p) const
    {
        if (!(mean() > 0))
        {
            return 1 / (4 * M_PI);
        }

        return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
    }

    int depthAt(Point2 p) const
    {
        return m_nodes[0].depthAt(p, m_nodes);
    }

    int depth() const
    {
        return m_maxDepth;
    }

    Point2 sample(Sampler *sampler) const
    {
        if (!(mean() > 0))
        {
            return sampler->next2D();
        }

        Point2 res = m_nodes[0].sample(sampler, m_nodes);

        res.x = math::clamp(res.x, 0.0f, 1.0f);
        res.y = math::clamp(res.y, 0.0f, 1.0f);

        return res;
    }

    size_t numNodes() const
    {
        return m_nodes.size();
    }

    float statisticalWeight() const
    {
        return m_atomic.statisticalWeight;
    }

    void setStatisticalWeight(float statisticalWeight)
    {
        m_atomic.statisticalWeight = statisticalWeight;
    }

    void reset(const DTree &previousDTree, int newMaxDepth, float subdivisionThreshold)
    {
        m_atomic = Atomic{};
        m_maxDepth = 0;
        m_nodes.clear();
        m_nodes.emplace_back();

        struct StackNode
        {
            size_t nodeIndex;
            size_t otherNodeIndex;
            const DTree *otherDTree;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 0, &previousDTree, 1});

        const float total = previousDTree.m_atomic.sum;

        // Create the topology of the new DTree to be the refined version
        // of the previous DTree. Subdivision is recursive if enough energy is there.
        while (!nodeIndices.empty())
        {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            m_maxDepth = std::max(m_maxDepth, sNode.depth);

            for (int i = 0; i < 4; ++i)
            {
                const QuadTreeNode &otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
                const float fraction = total > 0 ? (otherNode.sum(i) / total) : std::pow(0.25f, sNode.depth);
                SAssert(fraction <= 1.0f + Epsilon);

                if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold)
                {
                    if (!otherNode.isLeaf(i))
                    {
                        SAssert(sNode.otherDTree == &previousDTree);
                        nodeIndices.push({m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1});
                    }
                    else
                    {
                        nodeIndices.push({m_nodes.size(), m_nodes.size(), this, sNode.depth + 1});
                    }

                    m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
                    m_nodes.emplace_back();
                    m_nodes.back().setSum(otherNode.sum(i) / 4);

                    if (m_nodes.size() > std::numeric_limits<uint16_t>::max())
                    {
                        SLog(EWarn, "DTreeWrapper hit maximum children count.");
                        nodeIndices = std::stack<StackNode>();
                        break;
                    }
                }
            }
        }

        // Uncomment once memory becomes an issue.
        // m_nodes.shrink_to_fit();

        for (auto &node : m_nodes)
        {
            node.setSum(0);
        }
    }

    size_t approxMemoryFootprint() const
    {
        return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
    }

    void build()
    {
        auto &root = m_nodes[0];

        // Build the quadtree recursively, starting from its root.
        root.build(m_nodes);

        // Ensure that the overall sum of irradiance estimates equals
        // the sum of irradiance estimates found in the quadtree.
        float sum = 0;
        for (int i = 0; i < 4; ++i)
        {
            sum += root.sum(i);
        }
        m_atomic.sum.store(sum);
    }

private:
    std::vector<QuadTreeNode> m_nodes;

    struct Atomic
    {
        Atomic()
        {
            sum.store(0, std::memory_order_relaxed);
            statisticalWeight.store(0, std::memory_order_relaxed);
        }

        Atomic(const Atomic &arg)
        {
            *this = arg;
        }

        Atomic &operator=(const Atomic &arg)
        {
            sum.store(arg.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
            statisticalWeight.store(arg.statisticalWeight.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }

        std::atomic<float> sum;
        std::atomic<float> statisticalWeight;

    } m_atomic;

    int m_maxDepth;
};

struct DTreeRecord
{
    Vector d;
    float radiance, product;
    float woPdf, bsdfPdf, dTreePdf;
    float statisticalWeight;
    bool isDelta;
};

Vector canonicalToDir(Point2 p)
{
    const float cosTheta = 2 * p.x - 1;
    const float phi = 2 * M_PI * p.y;

    const float sinTheta = sqrt(1 - cosTheta * cosTheta);
    float sinPhi, cosPhi;
    math::sincos(phi, &sinPhi, &cosPhi);

    return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}

Point2 dirToCanonical(const Vector &d)
{
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z))
    {
        return {0, 0};
    }

    const float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
    float phi = std::atan2(d.y, d.x);
    while (phi < 0)
        phi += 2.0 * M_PI;

    return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
}

struct DTreeWrapper
{
public:
    DTreeWrapper()
    {
    }

    void record(const DTreeRecord &rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss)
    {
        if (!rec.isDelta)
        {
            float irradiance = rec.radiance / rec.woPdf;
            building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, directionalFilter);
        }

        if (bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && rec.product > 0)
        {
            optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f);
        }
    }

    void build()
    {
        building.build();
        sampling = building;
    }

    void reset(int maxDepth, float subdivisionThreshold)
    {
        building.reset(sampling, maxDepth, subdivisionThreshold);
    }

    Vector sample(Sampler *sampler) const
    {
        return canonicalToDir(sampling.sample(sampler));
    }

    float pdf(const Vector &dir) const
    {
        return sampling.pdf(dirToCanonical(dir));
    }

    float pdfHistory(const Vector &dir, int version) const
    {
        return history[version].pdf(dirToCanonical(dir));
    }

    void archive()
    {
        history.push_back(sampling);
    }

    float diff(const DTreeWrapper &other) const
    {
        return 0.0f;
    }

    int depth() const
    {
        return sampling.depth();
    }

    size_t numNodes() const
    {
        return sampling.numNodes();
    }

    float meanRadiance() const
    {
        return sampling.mean();
    }

    float statisticalWeight() const
    {
        return sampling.statisticalWeight();
    }

    float statisticalWeightBuilding() const
    {
        return building.statisticalWeight();
    }

    void setStatisticalWeightBuilding(float statisticalWeight)
    {
        building.setStatisticalWeight(statisticalWeight);
    }

    size_t approxMemoryFootprint() const
    {
        // our amis only requires one dtree per snode
        // return sampling.approxMemoryFootprint();
        size_t ans = 0;
        for (auto i: history)
        {
            ans += i.approxMemoryFootprint();
        }
        return ans;
    }

    inline float bsdfSamplingFraction(float variable) const
    {
        return logistic(variable);
    }

    inline float dBsdfSamplingFraction_dVariable(float variable) const
    {
        float fraction = bsdfSamplingFraction(variable);
        return fraction * (1 - fraction);
    }

    inline float bsdfSamplingFraction() const
    {
        return bsdfSamplingFraction(bsdfSamplingFractionOptimizer.variable());
    }

    void optimizeBsdfSamplingFraction(const DTreeRecord &rec, float ratioPower)
    {
        m_lock.lock();

        // GRADIENT COMPUTATION
        float variable = bsdfSamplingFractionOptimizer.variable();
        float samplingFraction = bsdfSamplingFraction(variable);

        // Loss gradient w.r.t. sampling fraction
        float mixPdf = samplingFraction * rec.bsdfPdf + (1 - samplingFraction) * rec.dTreePdf;
        float ratio = std::pow(rec.product / mixPdf, ratioPower);
        float dLoss_dSamplingFraction = -ratio / rec.woPdf * (rec.bsdfPdf - rec.dTreePdf);

        // Chain rule to get loss gradient w.r.t. trainable variable
        float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

        // We want some regularization such that our parameter does not become too big.
        // We use l2 regularization, resulting in the following linear gradient.
        float l2RegGradient = 0.01f * variable;

        float lossGradient = l2RegGradient + dLoss_dVariable;

        // ADAM GRADIENT DESCENT
        bsdfSamplingFractionOptimizer.append(lossGradient, rec.statisticalWeight);

        m_lock.unlock();
    }

    void dump(BlobWriter &blob, const Point &p, const Vector &size) const
    {
        blob
            << (float)p.x << (float)p.y << (float)p.z
            << (float)size.x << (float)size.y << (float)size.z
            << (float)sampling.mean() << (uint64_t)sampling.statisticalWeight() << (uint64_t)sampling.numNodes();

        for (size_t i = 0; i < sampling.numNodes(); ++i)
        {
            const auto &node = sampling.node(i);
            for (int j = 0; j < 4; ++j)
            {
                blob << (float)node.sum(j) << (uint16_t)node.child(j);
            }
        }
    }

private:
    DTree building;
    DTree sampling;
    std::vector<DTree> history;

    AdamOptimizer bsdfSamplingFractionOptimizer{0.01f};

    class SpinLock
    {
    public:
        SpinLock()
        {
            m_mutex.clear(std::memory_order_release);
        }

        SpinLock(const SpinLock &other) { m_mutex.clear(std::memory_order_release); }
        SpinLock &operator=(const SpinLock &other) { return *this; }

        void lock()
        {
            while (m_mutex.test_and_set(std::memory_order_acquire))
            {
            }
        }

        void unlock()
        {
            m_mutex.clear(std::memory_order_release);
        }

    private:
        std::atomic_flag m_mutex;
    } m_lock;
};

struct STreeNode
{
    STreeNode()
    {
        children = {};
        isLeaf = true;
        axis = 0;
    }

    int childIndex(Point &p) const
    {
        if (p[axis] < 0.5f)
        {
            p[axis] *= 2;
            return 0;
        }
        else
        {
            p[axis] = (p[axis] - 0.5f) * 2;
            return 1;
        }
    }

    int nodeIndex(Point &p) const
    {
        return children[childIndex(p)];
    }

    DTreeWrapper *dTreeWrapper(Point &p, Vector &size, std::vector<STreeNode> &nodes)
    {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf)
        {
            return &dTree;
        }
        else
        {
            size[axis] /= 2;
            return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
        }
    }

    const DTreeWrapper *dTreeWrapper() const
    {
        return &dTree;
    }

    int depth(Point &p, const std::vector<STreeNode> &nodes) const
    {
        SAssert(p[axis] >= 0 && p[axis] <= 1);
        if (isLeaf)
        {
            return 1;
        }
        else
        {
            return 1 + nodes[nodeIndex(p)].depth(p, nodes);
        }
    }

    int depth(const std::vector<STreeNode> &nodes) const
    {
        int result = 1;

        if (!isLeaf)
        {
            for (auto c : children)
            {
                result = std::max(result, 1 + nodes[c].depth(nodes));
            }
        }

        return result;
    }

    void forEachLeaf(
        std::function<void(const DTreeWrapper *, const Point &, const Vector &)> func,
        Point p, Vector size, const std::vector<STreeNode> &nodes) const
    {

        if (isLeaf)
        {
            func(&dTree, p, size);
        }
        else
        {
            size[axis] /= 2;
            for (int i = 0; i < 2; ++i)
            {
                Point childP = p;
                if (i == 1)
                {
                    childP[axis] += size[axis];
                }

                nodes[children[i]].forEachLeaf(func, childP, size, nodes);
            }
        }
    }

    float computeOverlappingVolume(const Point &min1, const Point &max1, const Point &min2, const Point &max2)
    {
        float lengths[3];
        for (int i = 0; i < 3; ++i)
        {
            lengths[i] = std::max(std::min(max1[i], max2[i]) - std::max(min1[i], min2[i]), 0.0f);
        }
        return lengths[0] * lengths[1] * lengths[2];
    }

    void record(const Point &min1, const Point &max1, Point min2, Vector size2, const DTreeRecord &rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, std::vector<STreeNode> &nodes)
    {
        float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
        if (w > 0)
        {
            if (isLeaf)
            {
                dTree.record({rec.d, rec.radiance, rec.product, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.statisticalWeight * w, rec.isDelta}, directionalFilter, bsdfSamplingFractionLoss);
            }
            else
            {
                size2[axis] /= 2;
                for (int i = 0; i < 2; ++i)
                {
                    if (i & 1)
                    {
                        min2[axis] += size2[axis];
                    }

                    nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes);
                }
            }
        }
    }

    bool isLeaf;
    DTreeWrapper dTree;
    int axis;
    std::array<uint32_t, 2> children;
};

class STree
{
public:
    STree(const AABB &aabb)
    {
        clear();

        m_aabb = aabb;

        // Enlarge AABB to turn it into a cube. This has the effect
        // of nicer hierarchical subdivisions.
        Vector size = m_aabb.max - m_aabb.min;
        float maxSize = std::max(std::max(size.x, size.y), size.z);
        m_aabb.max = m_aabb.min + Vector(maxSize);
    }

    size_t mem() const
    {
        size_t approxMemoryFootprint = 0;
        for (const auto &node : m_nodes)
        {
            approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
        }
        return approxMemoryFootprint;
    }

    void clear()
    {
        m_nodes.clear();
        m_nodes.emplace_back();
    }

    void subdivideAll()
    {
        int nNodes = (int)m_nodes.size();
        for (int i = 0; i < nNodes; ++i)
        {
            if (m_nodes[i].isLeaf)
            {
                subdivide(i, m_nodes);
            }
        }
    }

    void subdivide(int nodeIdx, std::vector<STreeNode> &nodes)
    {
        // Add 2 child nodes
        nodes.resize(nodes.size() + 2);

        if (nodes.size() > std::numeric_limits<uint32_t>::max())
        {
            SLog(EWarn, "DTreeWrapper hit maximum children count.");
            return;
        }

        STreeNode &cur = nodes[nodeIdx];
        for (int i = 0; i < 2; ++i)
        {
            uint32_t idx = (uint32_t)nodes.size() - 2 + i;
            cur.children[i] = idx;
            nodes[idx].axis = (cur.axis + 1) % 3;
            nodes[idx].dTree = cur.dTree;
            nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() / 2);
        }
        cur.isLeaf = false;
        cur.dTree = {}; // Reset to an empty dtree to save memory.
    }

    DTreeWrapper *dTreeWrapper(Point p, Vector &size)
    {
        size = m_aabb.getExtents();
        p = Point(p - m_aabb.min);
        p.x /= size.x;
        p.y /= size.y;
        p.z /= size.z;

        return m_nodes[0].dTreeWrapper(p, size, m_nodes);
    }

    DTreeWrapper *dTreeWrapper(Point p)
    {
        Vector size;
        return dTreeWrapper(p, size);
    }

    void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper *)> func) const
    {
        for (auto &node : m_nodes)
        {
            if (node.isLeaf)
            {
                func(&node.dTree);
            }
        }
    }

    void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper *, const Point &, const Vector &)> func) const
    {
        m_nodes[0].forEachLeaf(func, m_aabb.min, m_aabb.max - m_aabb.min, m_nodes);
    }

    void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper *)> func)
    {
        int nDTreeWrappers = static_cast<int>(m_nodes.size());

#pragma omp parallel for
        for (int i = 0; i < nDTreeWrappers; ++i)
        {
            if (m_nodes[i].isLeaf)
            {
                func(&m_nodes[i].dTree);
            }
        }
    }

    void record(const Point &p, const Vector &dTreeVoxelSize, DTreeRecord rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss)
    {
        float volume = 1;
        for (int i = 0; i < 3; ++i)
        {
            volume *= dTreeVoxelSize[i];
        }

        rec.statisticalWeight /= volume;
        m_nodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, m_aabb.min, m_aabb.getExtents(), rec, directionalFilter, bsdfSamplingFractionLoss, m_nodes);
    }

    void dump(BlobWriter &blob) const
    {
        forEachDTreeWrapperConstP([&blob](const DTreeWrapper *dTree, const Point &p, const Vector &size)
                                  {
            if (dTree->statisticalWeight() > 0) {
                dTree->dump(blob, p, size);
            } });
    }

    bool shallSplit(const STreeNode &node, int depth, size_t samplesRequired)
    {
        return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.statisticalWeightBuilding() > samplesRequired;
    }

    void refine(size_t sTreeThreshold, int maxMB)
    {
        if (maxMB >= 0)
        {
            size_t approxMemoryFootprint = 0;
            for (const auto &node : m_nodes)
            {
                approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
            }

            if (approxMemoryFootprint / 1000000 >= (size_t)maxMB)
            {
                return;
            }
        }

        struct StackNode
        {
            size_t index;
            int depth;
        };

        std::stack<StackNode> nodeIndices;
        nodeIndices.push({0, 1});
        while (!nodeIndices.empty())
        {
            StackNode sNode = nodeIndices.top();
            nodeIndices.pop();

            // Subdivide if needed and leaf
            if (m_nodes[sNode.index].isLeaf)
            {
                if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold))
                {
                    subdivide((int)sNode.index, m_nodes);
                }
            }

            // Add children to stack if we're not
            if (!m_nodes[sNode.index].isLeaf)
            {
                const STreeNode &node = m_nodes[sNode.index];
                for (int i = 0; i < 2; ++i)
                {
                    nodeIndices.push({node.children[i], sNode.depth + 1});
                }
            }
        }

        // Uncomment once memory becomes an issue.
        // m_nodes.shrink_to_fit();
    }

    const AABB &aabb() const
    {
        return m_aabb;
    }

private:
    std::vector<STreeNode> m_nodes;
    AABB m_aabb;
};

static StatsCounter avgPathLength("Guided path tracer", "Average path length", EAverage);

class GuidedPathTracerAMISPathspace : public MonteCarloIntegrator
{
public:
    GuidedPathTracerAMISPathspace(const Properties &props) : MonteCarloIntegrator(props)
    {
        m_neeStr = props.getString("nee", "never");
        if (m_neeStr == "never")
        {
            m_nee = ENever;
        }
        else if (m_neeStr == "kickstart")
        {
            m_nee = EKickstart;
        }
        else if (m_neeStr == "always")
        {
            m_nee = EAlways;
        }
        else
        {
            Assert(false);
        }

        m_sampleCombinationStr = props.getString("sampleCombination", "automatic");
        if (m_sampleCombinationStr == "amis")
        {
            m_sampleCombination = ESampleCombination::EAMIS;
        }
        else
        {
            Assert(false);
        }

        m_spatialFilterStr = props.getString("spatialFilter", "nearest");
        if (m_spatialFilterStr == "nearest")
        {
            m_spatialFilter = ESpatialFilter::ENearest;
        }
        else if (m_spatialFilterStr == "stochastic")
        {
            m_spatialFilter = ESpatialFilter::EStochasticBox;
        }
        else if (m_spatialFilterStr == "box")
        {
            m_spatialFilter = ESpatialFilter::EBox;
        }
        else
        {
            Assert(false);
        }

        m_directionalFilterStr = props.getString("directionalFilter", "nearest");
        if (m_directionalFilterStr == "nearest")
        {
            m_directionalFilter = EDirectionalFilter::ENearest;
        }
        else if (m_directionalFilterStr == "box")
        {
            m_directionalFilter = EDirectionalFilter::EBox;
        }
        else
        {
            Assert(false);
        }

        m_bsdfSamplingFractionLossStr = props.getString("bsdfSamplingFractionLoss", "none");
        if (m_bsdfSamplingFractionLossStr == "none")
        {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::ENone;
        }
        else if (m_bsdfSamplingFractionLossStr == "kl")
        {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EKL;
        }
        else if (m_bsdfSamplingFractionLossStr == "var")
        {
            m_bsdfSamplingFractionLoss = EBsdfSamplingFractionLoss::EVariance;
        }
        else
        {
            Assert(false);
        }

        m_sdTreeMaxMemory = props.getInteger("sdTreeMaxMemory", -1);
        m_sTreeThreshold = props.getInteger("sTreeThreshold", 4000);
        m_dTreeThreshold = props.getFloat("dTreeThreshold", 0.01f);
        m_bsdfSamplingFraction = props.getFloat("bsdfSamplingFraction", 0.5f);
        m_sppPerPass = props.getInteger("sppPerPass", 4);

        m_budgetStr = props.getString("budgetType", "seconds");
        if (m_budgetStr == "spp")
        {
            m_budgetType = ESpp;
        }
        else if (m_budgetStr == "seconds")
        {
            m_budgetType = ESeconds;
        }
        else
        {
            Assert(false);
        }

        m_sampleAllocSeqStr = props.getString("sampleAllocSeq", "double");
        if (m_sampleAllocSeqStr == "double")
        {
            m_sampleAllocSeq = ESampleAllocSeq::EDouble;
        }
        else if (m_sampleAllocSeqStr == "halfdouble")
        {
            m_sampleAllocSeq = ESampleAllocSeq::EHalfdouble;
        }
        else if (m_sampleAllocSeqStr == "uniform")
        {
            m_sampleAllocSeq = ESampleAllocSeq::EUniform;
        }
        else
        {
            Assert(false);
        }
        m_budget = props.getFloat("budget", 300.0f);

        m_dumpSDTree = props.getBoolean("dumpSDTree", false);
        m_tempParam = props.getFloat("tempParam", 0);
        g_tempParam = m_tempParam;
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
                                         RenderQueue *queue, const RenderJob *job,
                                         int sceneResID, int sensorResID, int samplerResID, int integratorResID)
    {

        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
                                                                  queue, scene->getBlockSize());

        proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    void resetSDTree()
    {
        Log(EInfo, "Resetting distributions for sampling.");

        int iter = m_iter;
        int t_iter = m_iter;
        if (m_sampleAllocSeq == ESampleAllocSeq::EHalfdouble)
        {
            iter = std::max(0, iter - 4);
        }
        HDTimer t1;
        if (t_iter > 0) m_sdTree->forEachDTreeWrapperParallel([this, t_iter](DTreeWrapper *dTree)
                                              {  dTree->archive(); });
        statsAMISTimeArchive += t1.value();
        m_sdTree->refine((size_t)(std::sqrt(std::pow(2, iter) * m_sppPerPass / 4) * m_sTreeThreshold), m_sdTreeMaxMemory);
        m_sdTree->forEachDTreeWrapperParallel([this](DTreeWrapper *dTree)
                                              { dTree->reset(20, m_dTreeThreshold); });
    }

    void buildSDTree()
    {
        Log(EInfo, "Building distributions for sampling.");

        // Build distributions
        m_sdTree->forEachDTreeWrapperParallel([](DTreeWrapper *dTree)
                                              { dTree->build(); });

        // Gather statistics
        int maxDepth = 0;
        int minDepth = std::numeric_limits<int>::max();
        float avgDepth = 0;
        float maxAvgRadiance = 0;
        float minAvgRadiance = std::numeric_limits<float>::max();
        float avgAvgRadiance = 0;
        size_t maxNodes = 0;
        size_t minNodes = std::numeric_limits<size_t>::max();
        float avgNodes = 0;
        float maxStatisticalWeight = 0;
        float minStatisticalWeight = std::numeric_limits<float>::max();
        float avgStatisticalWeight = 0;

        int nPoints = 0;
        int nPointsNodes = 0;

        m_sdTree->forEachDTreeWrapperConst([&](const DTreeWrapper *dTree)
                                           {
            const int depth = dTree->depth();
            maxDepth = std::max(maxDepth, depth);
            minDepth = std::min(minDepth, depth);
            avgDepth += depth;

            const float avgRadiance = dTree->meanRadiance();
            maxAvgRadiance = std::max(maxAvgRadiance, avgRadiance);
            minAvgRadiance = std::min(minAvgRadiance, avgRadiance);
            avgAvgRadiance += avgRadiance;

            if (dTree->numNodes() > 1) {
                const size_t nodes = dTree->numNodes();
                maxNodes = std::max(maxNodes, nodes);
                minNodes = std::min(minNodes, nodes);
                avgNodes += nodes;
                ++nPointsNodes;
            }

            const float statisticalWeight = dTree->statisticalWeight();
            maxStatisticalWeight = std::max(maxStatisticalWeight, statisticalWeight);
            minStatisticalWeight = std::min(minStatisticalWeight, statisticalWeight);
            avgStatisticalWeight += statisticalWeight;

            ++nPoints; });

        if (nPoints > 0)
        {
            avgDepth /= nPoints;
            avgAvgRadiance /= nPoints;

            if (nPointsNodes > 0)
            {
                avgNodes /= nPointsNodes;
            }

            avgStatisticalWeight /= nPoints;
        }

        Log(EInfo,
            "Distribution statistics:\n"
            "  Depth         = [%d, %f, %d]\n"
            "  Mean radiance = [%f, %f, %f]\n"
            "  Node count    = [" SIZE_T_FMT ", %f, " SIZE_T_FMT "]\n"
            "  Stat. weight  = [%f, %f, %f]\n",
            minDepth, avgDepth, maxDepth,
            minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
            minNodes, avgNodes, maxNodes,
            minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight);

        m_isBuilt = true;
    }

    void dumpSDTree(Scene *scene, ref<Sensor> sensor)
    {
        std::ostringstream extension;
        extension << "-" << std::setfill('0') << std::setw(2) << m_iter << ".sdt";
        fs::path path = scene->getDestinationFile();
        auto cameraMatrix = sensor->getWorldTransform()->eval(0).getMatrix();

        BlobWriter blob(path.string());

        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                blob << (float)cameraMatrix(i, j);
            }
        }

        m_sdTree->dump(blob);
    }

    bool performRenderPasses(float &variance, int numPasses, Scene *scene, RenderQueue *queue, const RenderJob *job,
                             int sceneResID, int sensorResID, int samplerResID, int integratorResID)
    {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        g_sensor = sensor;
        ref<Film> film = sensor->getFilm();

        // m_image->clear(); // ! we do not clear to accumulate ? is this necessary now?
        m_squaredImage->clear();

        size_t totalBlocks = 0;

        Log(EInfo, "Rendering %d render passes.", numPasses);

        int N = numPasses * m_sppPerPass;
        m_sampleCounts.push_back(N);

        auto start = std::chrono::steady_clock::now();

        HDTimer timer_phase_renderpass;
        for (int i = 0; i < numPasses; ++i)
        {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            m_renderProcesses.push_back(process);
            totalBlocks += process->totalBlocks();
        }

        bool result = true;
        int passesRenderedLocal = 0;

        static const size_t processBatchSize = 128;

        for (size_t i = 0; i < m_renderProcesses.size(); i += processBatchSize)
        {
            const size_t start = i;
            const size_t end = std::min(i + processBatchSize, m_renderProcesses.size());
            for (size_t j = start; j < end; ++j)
            {
                sched->schedule(m_renderProcesses[j]);
            }

            for (size_t j = start; j < end; ++j)
            {
                auto &process = m_renderProcesses[j];
                sched->wait(process);

                ++m_passesRendered;
                ++m_passesRenderedThisIter;
                ++passesRenderedLocal;

                int progress = 0;
                bool shouldAbort;
                switch (m_budgetType)
                {
                case ESpp:
                    progress = m_passesRendered;
                    shouldAbort = false;
                    break;
                case ESeconds:
                    progress = (int)computeElapsedSeconds(m_startTime);
                    shouldAbort = progress > m_budget;
                    break;
                default:
                    Assert(false);
                    break;
                }

                m_progress->update(progress);

                if (process->getReturnStatus() != ParallelProcess::ESuccess)
                {
                    result = false;
                    shouldAbort = true;
                }

                if (shouldAbort)
                {
                    goto l_abort;
                }
            }
        }
    l_abort:

        for (auto &process : m_renderProcesses)
        {
            sched->cancel(process);
        }

        std::cout << "all mem " << m_sdTree->mem() * 1.0 / 1048576 << "MB" << std::endl;
        m_renderProcesses.clear();

        variance = 0;

        float seconds = computeElapsedSeconds(start);

        Log(EInfo, "%.2f seconds, Total passes: %d",
            seconds, m_passesRendered);

        return result;
    }

    bool doNeeWithSpp(int spp)
    {
        switch (m_nee)
        {
        case ENever:
            return false;
        case EKickstart:
            return spp < 128;
        default:
            return true;
        }
    }

    bool renderSPP(Scene *scene, RenderQueue *queue, const RenderJob *job,
                   int sceneResID, int sensorResID, int samplerResID, int integratorResID)
    {
        memset(len_counts, 0, sizeof(len_counts));

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t sampleCount = (size_t)m_budget;

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        int nPasses = (int)std::ceil(sampleCount / (float)m_sppPerPass);
        sampleCount = m_sppPerPass * nPasses;

        g_sampleCount = sampleCount;

        bool result = true;
        float currentVarAtEnd = std::numeric_limits<float>::infinity();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", nPasses, job));

        while (result && m_passesRendered < nPasses)
        {
            HDTimer timer_phase_total;
            const int sppRendered = m_passesRendered * m_sppPerPass;
            m_doNee = doNeeWithSpp(sppRendered);

            int remainingPasses = nPasses - m_passesRendered;
            int passesThisIteration = std::min(remainingPasses, 1 << m_iter); // ! this line is modified from the original code release

            // If the next iteration does not manage to double the number of passes once more
            // then it would be unwise to throw away the current iteration. Instead, extend
            // the current iteration to the end.
            // This condition can also be interpreted as: the last iteration must always use
            // at _least_ half the total sample budget.
            if (remainingPasses - passesThisIteration < 2 * passesThisIteration)
            {
                passesThisIteration = remainingPasses;
            }
            if (m_sampleAllocSeq == ESampleAllocSeq::EHalfdouble)
            {
                passesThisIteration = 1 << std::max(0, m_iter - 4);
            }

            if (m_sampleAllocSeq == ESampleAllocSeq::EUniform)
            {
                passesThisIteration = 1;
            }
            if (remainingPasses - passesThisIteration < 0)
            {
                passesThisIteration = remainingPasses;
            }

            Log(EInfo, "ITERATION %d, %d passes", m_iter, passesThisIteration);

            g_passesThisIteration = passesThisIteration;

            m_isFinalIter = passesThisIteration >= remainingPasses;

            film->clear();

            // if ((m_sampleAllocSeq == ESampleAllocSeq::EUniform && (m_iter + 1) == std::pow(2, int(std::log2(m_iter + 1))))
            //     || (m_sampleAllocSeq == ESampleAllocSeq::EHalfdouble && (m_iter == 0 || m_iter % 2 == 1))
            //     || m_sampleAllocSeq == ESampleAllocSeq::EDouble)
            resetSDTree();

            float variance;
            if (!performRenderPasses(variance, passesThisIteration, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID))
            {
                result = false;
                break;
            }

            const float lastVarAtEnd = currentVarAtEnd;
            currentVarAtEnd = passesThisIteration * variance / remainingPasses;

            Log(EInfo,
                "Extrapolated var:\n"
                "  Last:    %f\n"
                "  Current: %f\n",
                lastVarAtEnd, currentVarAtEnd);

            remainingPasses -= passesThisIteration;
            buildSDTree();

            ++m_iter;
            m_passesRenderedThisIter = 0;
        }

        return result;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
                    int sceneResID, int sensorResID, int samplerResID, int integratorResID)
    {
        std::cout << "not supported" << std::endl;
        std::cerr << "not supported" << std::endl;
        exit(1);
        return false;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
                int sceneResID, int sensorResID, int samplerResID)
    {

        m_sdTree = std::shared_ptr<STree>(new STree(scene->getAABB()));
        m_iter = 0;
        m_isFinalIter = false;

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_film = film;

        auto properties = Properties("hdrfilm");
        properties.setInteger("width", film->getSize().x);
        properties.setInteger("height", film->getSize().y);
        m_varianceBuffer = static_cast<Film *>(PluginManager::getInstance()->createObject(MTS_CLASS(Film), properties));
        m_varianceBuffer->setDestinationFile(scene->getDestinationFile(), 0);

        m_squaredImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());
        m_image = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize());

        m_images.clear();
        m_variances.clear();
        m_sampleCounts.clear();

        m_amisImage = new ImageBlock(Bitmap::ESpectrumAlphaWeight, film->getCropSize(), film->getReconstructionFilter());
        m_amisImage->clear();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        m_startTime = std::chrono::steady_clock::now();

        m_passesRendered = 0;
        switch (m_budgetType)
        {
        case ESpp:
            result = renderSPP(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            break;
        case ESeconds:
            result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            break;
        default:
            Assert(false);
            break;
        }

        sched->unregisterResource(integratorResID);

        m_progress = nullptr;

        fuseImageSamples(film, m_sppPerPass);
        printMystats();

        return result;
    }

    void fuseImageSamples(ref<Film> film, int sppPerPass)
    {
        Log(EInfo, "fuseImageSamples begin");
        HDTimer timer;
        amisSplatSamples();
        amisSplatPostproc();
        m_sppPerPass = sppPerPass;
        Log(EInfo, "fuseImageSamples end, use %.3f sec", timer.value());
    }

    void renderBlock(const Scene *scene, const Sensor *sensor,
                     Sampler *sampler, ImageBlock *block, const bool &stop,
                     const std::vector<TPoint2<uint8_t>> &points) const
    {

        HDTimer timer;

        float diffScaleFactor = 1.0f /
                                std::sqrt((float)m_sppPerPass);

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        for (size_t i = 0; i < points.size(); ++i)
        {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset());
            if (stop)
                break;

            for (int j = 0; j < m_sppPerPass; j++)
            {
                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                if (i + j == 0) g_first_vertex = sensorRay.o;

                sensorRay.scaleDifferential(diffScaleFactor);

                rRec.samplePos = samplePos;

                auto L = Li(sensorRay, rRec);
                sampler->advance();
            }
        }

        m_image->put(block);
    }

    void cancel()
    {
        const auto &scheduler = Scheduler::getInstance();
        for (size_t i = 0; i < m_renderProcesses.size(); ++i)
        {
            scheduler->cancel(m_renderProcesses[i]);
        }
    }

    Spectrum sampleMat(const BSDF *bsdf, BSDFSamplingRecord &bRec, float &woPdf, float &bsdfPdf, float &dTreePdf, float bsdfSamplingFraction, RadianceQueryRecord &rRec, const DTreeWrapper *dTree) const
    {
        Point2 sample = rRec.nextSample2D();

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll))
        {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            dTreePdf = 0;
            return result;
        }

        Spectrum result;
        if (sample.x < bsdfSamplingFraction)
        {
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero())
            {
                woPdf = bsdfPdf = dTreePdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta)
            {
                dTreePdf = 0;
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction;
            }

            result *= bsdfPdf;
        }
        else
        {
            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);
            bRec.wo = bRec.its.toLocal(dTree->sample(rRec.sampler));
            result = bsdf->eval(bRec);
        }

        pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);
        if (woPdf == 0)
        {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    void pdfMat(float &woPdf, float &bsdfPdf, float &dTreePdf, float bsdfSamplingFraction, const BSDF *bsdf, const BSDFSamplingRecord &bRec, const DTreeWrapper *dTree) const
    {
        dTreePdf = 0;

        auto type = bsdf->getType();
        if (!m_isBuilt || !dTree || (type & BSDF::EDelta) == (type & BSDF::EAll))
        {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        if (!std::isfinite(bsdfPdf))
        {
            woPdf = 0;
            return;
        }

        dTreePdf = dTree->pdf(bRec.its.toWorld(bRec.wo));
        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * dTreePdf;
    }

    struct Vertex
    {
        DTreeWrapper *dTree;
        Vector dTreeVoxelSize;
        Ray ray;

        Spectrum throughput;
        Spectrum bsdfVal;

        Spectrum radiance;

        float woPdf, bsdfPdf, dTreePdf;
        bool isDelta;

        void record(const Spectrum &r)
        {
            radiance += r;
        }

        void commit(STree &sdTree, float statisticalWeight, ESpatialFilter spatialFilter, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, Sampler *sampler)
        {
            if (!(woPdf > 0) || !radiance.isValid() || !bsdfVal.isValid())
            {
                return;
            }

            Spectrum localRadiance = Spectrum{0.0f};
            if (throughput[0] * woPdf > Epsilon)
                localRadiance[0] = radiance[0] / throughput[0];
            if (throughput[1] * woPdf > Epsilon)
                localRadiance[1] = radiance[1] / throughput[1];
            if (throughput[2] * woPdf > Epsilon)
                localRadiance[2] = radiance[2] / throughput[2];
            Spectrum product = localRadiance * bsdfVal;

            DTreeRecord rec{ray.d, localRadiance.average(), product.average(), woPdf, bsdfPdf, dTreePdf, statisticalWeight, isDelta};
            switch (spatialFilter)
            {
            case ESpatialFilter::ENearest:
                dTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                break;
            case ESpatialFilter::EStochasticBox:
            {
                DTreeWrapper *splatDTree = dTree;

                // Jitter the actual position within the
                // filter box to perform stochastic filtering.
                Vector offset = dTreeVoxelSize;
                offset.x *= sampler->next1D() - 0.5f;
                offset.y *= sampler->next1D() - 0.5f;
                offset.z *= sampler->next1D() - 0.5f;

                Point origin = sdTree.aabb().clip2(ray.o + offset);
                splatDTree = sdTree.dTreeWrapper(origin);
                if (splatDTree)
                {
                    splatDTree->record(rec, directionalFilter, bsdfSamplingFractionLoss);
                }
                break;
            }
            case ESpatialFilter::EBox:
                sdTree.record(ray.o, dTreeVoxelSize, rec, directionalFilter, bsdfSamplingFractionLoss);
                break;
            }
        }
    };

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        auto samplePos = rRec.samplePos;
        Vector3f last_wo(0.0f);

        static const int MAX_NUM_VERTICES = 32;
        std::array<Vertex, MAX_NUM_VERTICES> vertices;

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        MediumSamplingRecord mRec;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        float eta = 1.0f;
        auto p0 = r.o + r.d;

        /* Perform the first ray intersection (or ignore if the
        intersection has already been provided). */
        rRec.rayIntersect(ray);

        Spectrum throughput(1.0f);
        bool scattered = false;

        float woPdf_product = 1.0f;
        float bsdfPdf_product = 1.0f;
        int nVertices = 0;

        std::vector<Point4f> path;

        auto recordRadiance = [&](Spectrum radiance)
        {
            Li += radiance;
            for (int i = 0; i < nVertices; ++i)
            {
                vertices[i].record(radiance);
            }
        };

        float emitterRadiance = 0;
        bool pass_through_diffuse = false;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0)
        {

            /* ==================================================================== */
            /*                 Radiative Transfer Equation sampling                 */
            /* ==================================================================== */
            if (rRec.medium && rRec.medium->sampleDistance(Ray(ray, 0, its.t), mRec, rRec.sampler))
            {
               
            }
            else
            {
                /* Sample
                tau(x, y) (Surface integral). This happens with probability mRec.pdfFailure
                Account for this and multiply by the proper per-color-channel transmittance.
                */
                if (rRec.medium)
                    throughput *= mRec.transmittance / mRec.pdfFailure;

                if (!its.isValid())
                {
                    /* If no intersection could be found, possibly return
                    attenuated radiance from a background luminaire */
                    if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    {
                        Spectrum value = throughput * scene->evalEnvironment(ray);
                        if (rRec.medium)
                            value *= rRec.medium->evalTransmittance(ray, rRec.sampler);
                        recordRadiance(value);
                    }

                    break;
                }

                /* Possibly include emitted radiance if requested */
                if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered)) 
                    recordRadiance(throughput * its.Le(-ray.d));

                /* Include radiance from a subsurface integrator if requested */
                if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                    recordRadiance(throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth));

                if (rRec.depth >= m_maxDepth && m_maxDepth != -1)
                    break;

                /* Prevent light leaks due to the use of shading normals */
                float wiDotGeoN = -dot(its.geoFrame.n, ray.d),
                      wiDotShN = Frame::cosTheta(its.wi);
                if (wiDotGeoN * wiDotShN < 0 && m_strictNormals)
                    break;

                const BSDF *bsdf = its.getBSDF();

                Vector dTreeVoxelSize;
                DTreeWrapper *dTree = nullptr;

                // We only guide smooth BRDFs for now. Analytic product sampling
                // would be conceivable for discrete decisions such as refraction vs
                // reflection.
                if (bsdf->getType() & BSDF::ESmooth)
                {
                    if (!its.isEmitter()) pass_through_diffuse = true;
                    dTree = m_sdTree->dTreeWrapper(its.p, dTreeVoxelSize);
                }

                float bsdfSamplingFraction = m_bsdfSamplingFraction;
                if (dTree && m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone)
                {
                    bsdfSamplingFraction = dTree->bsdfSamplingFraction();
                }

                /* ==================================================================== */
                /*                            BSDF sampling                             */
                /* ==================================================================== */

                /* Sample BSDF * cos(theta) */
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                float woPdf, bsdfPdf, dTreePdf;
                Spectrum bsdfWeight = sampleMat(bsdf, bRec, woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, rRec, dTree);

                /* ==================================================================== */
                /*                          Luminaire sampling                          */
                /* ==================================================================== */

                DirectSamplingRecord dRec(its);

                /* Estimate the direct illumination if this is requested */
                if (m_doNee &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                    (bsdf->getType() & BSDF::ESmooth))
                {
                    int interactions = m_maxDepth - rRec.depth - 1;

                    Spectrum value = scene->sampleAttenuatedEmitterDirect(
                        dRec, its, rRec.medium, interactions,
                        rRec.nextSample2D(), rRec.sampler);

                    if (!value.isZero())
                    {
                        BSDFSamplingRecord bRec(its, its.toLocal(dRec.d));

                        float woDotGeoN = dot(its.geoFrame.n, dRec.d);

                        /* Prevent light leaks due to the use of shading normals */
                        if (!m_strictNormals || woDotGeoN * Frame::cosTheta(bRec.wo) > 0)
                        {
                            /* Evaluate BSDF * cos(theta) */
                            const Spectrum bsdfVal = bsdf->eval(bRec);

                            /* Calculate prob. of having generated that direction using BSDF sampling */
                            const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                            float woPdf = 0, bsdfPdf = 0, dTreePdf = 0;
                            if (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            {
                                pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, dTree);
                            }

                            /* Weight using the power heuristic */
                            const float weight = miWeight(dRec.pdf, woPdf);

                            value *= bsdfVal;
                            Spectrum L = throughput * value * weight;

                            if (!m_isFinalIter && m_nee != EAlways)
                            {
                                if (dTree)
                                {
                                    Vertex v = Vertex{
                                        dTree,
                                        dTreeVoxelSize,
                                        Ray(its.p, dRec.d, 0),
                                        throughput * bsdfVal / dRec.pdf,
                                        bsdfVal,
                                        L,
                                        dRec.pdf,
                                        bsdfPdf,
                                        dTreePdf,
                                        false,
                                    };

                                    v.commit(*m_sdTree, 0.5f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
                                }
                            }

                            recordRadiance(L);
                        }
                    }
                }

                // BSDF handling
                if (bsdfWeight.isZero())
                    break;

                /* Prevent light leaks due to the use of shading normals */
                const Vector wo = its.toWorld(bRec.wo);
                float woDotGeoN = dot(its.geoFrame.n, wo);

                if (woDotGeoN * Frame::cosTheta(bRec.wo) <= 0 && m_strictNormals)
                    break;

                /* Trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);

                /* Keep track of the throughput, medium, and relative
                refractive index along the path */
                throughput *= bsdfWeight;
                woPdf_product *= woPdf;
                last_wo = its.toWorld(bRec.wo);
                auto type = bsdf->getType();
                path.push_back({its.p[0], its.p[1], its.p[2], (type & BSDF::EDelta) != (type & BSDF::EAll) ? bsdfPdf : -1});

                eta *= bRec.eta;
                if (its.isMediumTransition())
                    rRec.medium = its.getTargetMedium(ray.d);

                /* Handle index-matched medium transitions specially */
                if (bRec.sampledType == BSDF::ENull)
                {
                    if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                        break;

                    // There exist materials that are smooth/null hybrids (e.g. the mask BSDF), which means that
                    // for optimal-sampling-fraction optimization we need to record null transitions for such BSDFs.
                    if (m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter)
                    {
                        if (1 / woPdf > 0)
                        {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                Spectrum{0.0f},
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                true,
                            };

                            ++nVertices;
                        }
                    }

                    rRec.type = scattered ? RadianceQueryRecord::ERadianceNoEmission
                                          : RadianceQueryRecord::ERadiance;
                    scene->rayIntersect(ray, its);
                    rRec.depth++;
                    continue;
                }

                Spectrum value(0.0f);
                rayIntersectAndLookForEmitter(scene, rRec.sampler, rRec.medium,
                                              m_maxDepth - rRec.depth - 1, ray, its, dRec, value);

                /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
                if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)
                {
                    bool isDelta = bRec.sampledType & BSDF::EDelta;
                    const float emitterPdf = (m_doNee && !isDelta && !value.isZero()) ? scene->pdfEmitterDirect(dRec) : 0;

                    const float weight = miWeight(woPdf, emitterPdf);
                    Spectrum L = throughput * value * weight;
                    if (!L.isZero())
                    {
                        recordRadiance(L);
                    }

                    if ((!isDelta || m_bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone) && dTree && nVertices < MAX_NUM_VERTICES && !m_isFinalIter)
                    {
                        if (1 / woPdf > 0)
                        {
                            vertices[nVertices] = Vertex{
                                dTree,
                                dTreeVoxelSize,
                                ray,
                                throughput,
                                bsdfWeight * woPdf,
                                (m_nee == EAlways) ? Spectrum{0.0f} : L,
                                woPdf,
                                bsdfPdf,
                                dTreePdf,
                                isDelta,
                            };

                            ++nVertices;
                        }
                    }
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                /* Stop if indirect illumination was not requested */
                if (!(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;

                rRec.type = RadianceQueryRecord::ERadianceNoEmission;

                // Russian roulette
                if (rRec.depth++ >= m_rrDepth)
                {
                    float successProb = 1.0f;
                    if (dTree && !(bRec.sampledType & BSDF::EDelta))
                    {
                        if (!m_isBuilt)
                        {
                            successProb = throughput.max() * eta * eta;
                        }
                        else
                        {
                            // The adjoint russian roulette implementation of Mueller et al. [2017]
                            // was broken, effectively turning off russian roulette entirely.
                            // For reproducibility's sake, we therefore removed adjoint russian roulette
                            // from this codebase rather than fixing it.
                        }

                        successProb = std::max(0.1f, std::min(successProb, 0.99f));
                    }

                    if (rRec.nextSample1D() >= successProb)
                        break;
                    throughput /= successProb;
                }
            }

            scattered = true;
        }
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        // int splatToDistr = g_tempParam > 0 ? 1 - splatToFilm : 1;

        int splatToFilm = 1;
        int splatToDistr = 1;

        if (splatToDistr && nVertices > 0 && !m_isFinalIter)
        {
            for (int i = 0; i < nVertices; ++i)
            {
                vertices[i].commit(*m_sdTree, m_nee == EKickstart && m_doNee ? 0.5f : 1.0f, m_spatialFilter, m_directionalFilter, m_isBuilt ? m_bsdfSamplingFractionLoss : EBsdfSamplingFractionLoss::ENone, rRec.sampler);
            }
        }

        if (splatToFilm)
        {
            int thread_id = Thread::getThread()->getID();
            if (s_amisBufferThreadlocal[thread_id].size() == 0)
            {
            }
            float energy = Li[0] + Li[1] + Li[2];
            if (isnan(energy) || isinf(energy))
                energy = 0; 
            if (!m_isFinalIter)
                statsImageSamples++;
            bool contributed = energy > 0;
            if (contributed && !m_isFinalIter)
            {
                statsImageSamplesNonzero++;
            }
            if (contributed && path.size() > 0)
            {
                RawImageSample sample = {path, dirToCanonical(last_wo), Li, m_iter, energy};
                bool flag = true;
                if (!s_amisBufferThreadlocal[thread_id].empty() && statsRecordedVertices * 16.0 / 1048576 > std::abs(g_tempParam))
                {
                    if (sample.original_radiance < s_amisBufferThreadlocal[thread_id].top().original_radiance)
                    {
                        flag = false;
                    }
                }
                if (!m_isFinalIter && path.size() > 0 && pass_through_diffuse) {
                    if (flag) {
                        s_amisBufferThreadlocal[thread_id].push(sample);
                        statsImageSamplesAMIS++;
                        statsRecordedVertices += path.size() + 1; // the final iter is not counted since it can be optimized
                        while (statsRecordedVertices * 16.0 / 1048576 > std::abs(g_tempParam) && !s_amisBufferThreadlocal[thread_id].empty()) // number of cores
                        {
                            auto sample = s_amisBufferThreadlocal[thread_id].top();
                            s_amisBufferThreadlocal[thread_id].pop();
                            amisSplatOneSample(sample, g_tempParam < 0);
                            statsRecordedVertices -= sample.path.size() + 1; // the final iter is not counted since it can be optimized
                        }
                    }
                    else {
                        amisSplatOneSample(sample, g_tempParam < 0);
                    }
                }
                else {
                    if (!pass_through_diffuse || path.size() == 0   ) {
                        amisSplatOneSample(sample, false);
                    }
                    else 
                    {
                        amisSplatOneSample(sample, g_tempParam < 0 ? true : flag);
                    }
                }
            }
            else
            {
                m_amisImage->put(samplePos, Li, rRec.alpha);
            }
        }

        return Li;
    }

    /**
     * This function is called by the recursive ray tracing above after
     * having sampled a direction from a BSDF/phase function. Due to the
     * way in which this integrator deals with index-matched boundaries,
     * it is necessarily a bit complicated (though the improved performance
     * easily pays for the extra effort).
     *
     * This function
     *
     * 1. Intersects 'ray' against the scene geometry and returns the
     *    *first* intersection via the '_its' argument.
     *
     * 2. It checks whether the intersected shape was an emitter, or if
     *    the ray intersects nothing and there is an environment emitter.
     *    In this case, it returns the attenuated emittance, as well as
     *    a DirectSamplingRecord that can be used to query the hypothetical
     *    sampling density at the emitter.
     *
     * 3. If current shape is an index-matched medium transition, the
     *    integrator keeps on looking on whether a light source eventually
     *    follows after a potential chain of index-matched medium transitions,
     *    while respecting the specified 'maxDepth' limits. It then returns
     *    the attenuated emittance of this light source, while accounting for
     *    all attenuation that occurs on the wya.
     */
    void rayIntersectAndLookForEmitter(const Scene *scene, Sampler *sampler,
                                       const Medium *medium, int maxInteractions, Ray ray, Intersection &_its,
                                       DirectSamplingRecord &dRec, Spectrum &value) const
    {
        Intersection its2, *its = &_its;
        Spectrum transmittance(1.0f);
        bool surface = false;
        int interactions = 0;

        while (true)
        {
            surface = scene->rayIntersect(ray, *its);

            if (medium)
                transmittance *= medium->evalTransmittance(Ray(ray, 0, its->t), sampler);

            if (surface && (interactions == maxInteractions ||
                            !(its->getBSDF()->getType() & BSDF::ENull) ||
                            its->isEmitter()))
            {
                /* Encountered an occluder / light source */
                break;
            }

            if (!surface)
                break;

            if (transmittance.isZero())
                return;

            if (its->isMediumTransition())
                medium = its->getTargetMedium(ray.d);

            Vector wo = its->shFrame.toLocal(ray.d);
            BSDFSamplingRecord bRec(*its, -wo, wo, ERadiance);
            bRec.typeMask = BSDF::ENull;
            transmittance *= its->getBSDF()->eval(bRec, EDiscrete);

            ray.o = ray(its->t);
            ray.mint = Epsilon;
            its = &its2;

            if (++interactions > 100)
            { /// Just a precaution..
                Log(EWarn, "rayIntersectAndLookForEmitter(): round-off error issues?");
                return;
            }
        }

        if (surface)
        {
            /* Intersected something - check if it was a luminaire */
            if (its->isEmitter())
            {
                dRec.setQuery(ray, *its);
                value = transmittance * its->Le(-ray.d);
            }
        }
        else
        {
            /* Intersected nothing -- perhaps there is an environment map? */
            const Emitter *env = scene->getEnvironmentEmitter();

            if (env && env->fillDirectSamplingRecord(dRec, ray))
            {
                value = transmittance * env->evalEnvironment(RayDifferential(ray));
                dRec.dist = std::numeric_limits<float>::infinity();
                its->t = std::numeric_limits<float>::infinity();
            }
        }
    }

    float miWeight(float pdfA, float pdfB) const
    {
        // pdfA *= pdfA;
        // pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GuidedPathTracerAMISPathspace[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    double getAMISPdf(double bsdfPdf, const Point3f &pos, const Vector3f &dir, int iter, DTreeWrapper* dtw) const
    {
        double woPdf_film = 0.0f;
        if (iter > 0 && bsdfPdf > 0)
        {
            woPdf_film += bsdfPdf * 0.5;
            float dtpdf = dtw->pdfHistory(dir, iter - 1);
            if (isnan(dtpdf) || isinf(dtpdf))
            {
                dtpdf = 0;
            }
            woPdf_film += dtpdf * 0.5;
        }
        else
        {
            woPdf_film = bsdfPdf;
        }
        return woPdf_film;
    }


    mutable int len_counts[64];

    void amisSplatOneSample(const RawImageSample& sample, bool reweight = true) const
    {
        auto f3 = [](const Point4f& point) -> Point3f {
            return {point[0], point[1], point[2]};
        };
        
        int filmWidth = m_film->getSize().x, filmHeight = m_film->getSize().y;
        auto path = sample.path;
        Vector3f tt = normalize(f3(sample.path[0]) - g_first_vertex);
        Intersection its;
        PositionSamplingRecord psr(its);
        psr.time = 0;
        DirectionSamplingRecord dsr({tt.x, tt.y, tt.z});
        Point2 ps;
        bool fl = g_sensor->getSamplePosition(psr, dsr, ps);
        int x = ps[0], y = ps[1];
        int iter = sample.iter;
        x = std::max(0, x);
        x = std::min(x, filmWidth - 1);
        y = std::max(0, y);
        y = std::min(y, filmHeight - 1);

        Spectrum ans = sample.value;
        int n_iter = m_sampleCounts.size();
        if (reweight){
            float factor = 1.f;
            std::vector<float> pdfForEachIters(m_sampleCounts.size(), 1.0f);
            float pdfMixture = 0.f, denom = 0.f;
            len_counts[path.size()]++;
            for (int j = 0; j < path.size(); j++)
            {
                auto amisRec = path[j];
                auto dtw = m_sdTree->dTreeWrapper(f3(amisRec));
                int iter_start = 0, iter_end = m_sampleCounts.size();
                for (int iter = iter_start; iter < iter_end; iter++) {
                    Point3f dir_begin = f3(path[j]);
                    Point3f dir_end = j + 1 == path.size() ? dir_begin : f3(path[j + 1]);
                    Vector3f dir = j + 1 == path.size() ? canonicalToDir(sample.last_dir) : normalize(dir_end - dir_begin);
                    auto p = getAMISPdf(amisRec[3], f3(amisRec), dir, iter, dtw);
                    pdfForEachIters[iter] *= p;
                    if (iter == sample.iter) factor *= p;
                }
            }
            for (int iter = 0; iter < m_sampleCounts.size(); iter++)
            {
                float var_fac = 1;
                pdfMixture += pdfForEachIters[iter] * m_sampleCounts[iter] * var_fac;
                denom += m_sampleCounts[iter];
            }
            pdfMixture /= denom;
            ans *= factor / pdfMixture;
        }
        m_amisImage->put(Point2f(x + 0.5f, y + 0.5f), ans, 1.0f);
    }

    void amisSplatSamples()
    {
        auto *amisBufferPtr = &s_amisBuffer;
        auto &amisBuffer = *amisBufferPtr;

        HDTimer timer_splat;

        // merge buffers
        std::vector<std::priority_queue<RawImageSample> *> amisBufferPtrList;
        for (auto &[x, y] : s_amisBufferThreadlocal)
        {
            amisBufferPtrList.push_back(&y);
        }

        // splat
        std::cout << "begin splat " << std::endl;
#pragma omp parallel for
        for (int k = 0; k < amisBufferPtrList.size(); k++)
        {
            auto amisBufPtr = amisBufferPtrList[k];
            while(!amisBufPtr->empty())
            {
                auto sample = amisBufPtr->top(); 
                amisBufPtr->pop();
                amisSplatOneSample(sample);
            }
        }

        for(int i = 0; i < 64; i++)
        {
            if(len_counts[i] > 0)
            {
                std::cout << "len: " << i << " count: " << len_counts[i] << std::endl;
            }
        }
    }

    void amisSplatPostproc()
    {
        auto film = m_film;
        film->clear();
        ref<ImageBlock> imgBlockAMISImage = new ImageBlock(Bitmap::ESpectrum, film->getCropSize());
        ref<ImageBlock> imageBlockResidualImage = new ImageBlock(Bitmap::ESpectrum, film->getCropSize());
        int r = film->getReconstructionFilter()->getBorderSize();
        float coef = 1.0f;
        m_amisImage->getBitmap()->crop(Point2i(r, r), imgBlockAMISImage->getSize())->convert(imgBlockAMISImage->getBitmap(), coef);
        film->addBitmap(imgBlockAMISImage->getBitmap());
    }

private:
    /// The datastructure for guiding paths.
    std::shared_ptr<STree> m_sdTree;

    /// The squared values of our currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_squaredImage;
    /// The currently rendered image. Used to estimate variance.
    mutable ref<ImageBlock> m_image;

    std::vector<ref<Bitmap>> m_images;
    std::vector<float> m_variances;
    std::vector<int> m_sampleCounts;
    mutable float m_tempParam;
    /// This contains the currently estimated variance.
    mutable ref<Film> m_varianceBuffer;

    /// The modes of NEE which are supported.
    enum ENee
    {
        ENever,
        EKickstart,
        EAlways,
    };

    /**
        How to perform next event estimation (NEE). The following values are valid:
        - "never":     Never performs NEE.
        - "kickstart": Performs NEE for the first few iterations to initialize
                       the SDTree with good direct illumination estimates.
        - "always":    Always performs NEE.
        Default = "never"
    */
    std::string m_neeStr;
    ENee m_nee;

    /// Whether Li should currently perform NEE (automatically set during rendering based on m_nee).
    bool m_doNee;

    enum EBudget
    {
        ESpp,
        ESeconds,
    };

    /**
        What type of budget to use. The following values are valid:
        - "spp":     Budget is the number of samples per pixel.
        - "seconds": Budget is a time in seconds.
        Default = "seconds"
    */
    std::string m_budgetStr;
    EBudget m_budgetType;
    float m_budget;

    bool m_isBuilt = false;
    int m_iter;
    bool m_isFinalIter = false;

    int m_sppPerPass;

    int m_passesRendered;
    int m_passesRenderedThisIter;
    mutable std::unique_ptr<ProgressReporter> m_progress;

    std::vector<ref<BlockedRenderProcess>> m_renderProcesses;

    /**
        How to combine the samples from all path-guiding iterations:
        - "discard":    Discard all but the last iteration.
        - "automatic":  Discard all but the last iteration, but automatically assign an appropriately
                        larger budget to the last [Mueller et al. 2018].
        - "inversevar": Combine samples of the last 4 iterations based on their
                        mean pixel variance [Mueller et al. 2018].
        Default     = "automatic" (for reproducibility)
        Recommended = "inversevar"
    */
    std::string m_sampleCombinationStr;
    ESampleCombination m_sampleCombination;

    std::string m_sampleAllocSeqStr;
    ESampleAllocSeq m_sampleAllocSeq;

    /// Maximum memory footprint of the SDTree in MB. Stops subdividing once reached. -1 to disable.
    int m_sdTreeMaxMemory;

    /**
        The spatial filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "stochastic": Stochastic box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        - "box":        Box filter; improves the quality further at significant
                        additional computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "stochastic"
    */
    std::string m_spatialFilterStr;
    ESpatialFilter m_spatialFilter;

    /**
        The directional filter to use when splatting radiance samples into the SDTree.
        The following values are valid:
        - "nearest":    No filtering [Mueller et al. 2017].
        - "box":        Box filter; improves upon Mueller et al. [2017]
                        at nearly no computational cost.
        Default     = "nearest" (for reproducibility)
        Recommended = "box"
    */
    std::string m_directionalFilterStr;
    EDirectionalFilter m_directionalFilter;

    /**
        Leaf nodes of the spatial binary tree are subdivided if the number of samples
        they received in the last iteration exceeds c * sqrt(2^k) where c is this value
        and k is the iteration index. The first iteration has k==0.
        Default     = 12000 (for reproducibility)
        Recommended = 4000
    */
    int m_sTreeThreshold;

    /**
        Leaf nodes of the directional quadtree are subdivided if the fraction
        of energy they carry exceeds this value.
        Default = 0.01 (1%)
    */
    float m_dTreeThreshold;

    /**
        When guiding, we perform MIS with the balance heuristic between the guiding
        distribution and the BSDF, combined with probabilistically choosing one of the
        two sampling methods. This factor controls how often the BSDF is sampled
        vs. how often the guiding distribution is sampled.
        Default = 0.5 (50%)
    */
    float m_bsdfSamplingFraction;

    /**
        The loss function to use when learning the bsdfSamplingFraction using gradient
        descent, following the theory of Neural Importance Sampling [Mueller et al. 2018].
        The following values are valid:
        - "none":  No learning (uses the fixed `m_bsdfSamplingFraction`).
        - "kl":    Optimizes bsdfSamplingFraction w.r.t. the KL divergence.
        - "var":   Optimizes bsdfSamplingFraction w.r.t. variance.
        Default     = "none" (for reproducibility)
        Recommended = "kl"
    */
    std::string m_bsdfSamplingFractionLossStr;
    EBsdfSamplingFractionLoss m_bsdfSamplingFractionLoss;

    /**
        Whether to dump a binary representation of the SD-Tree to disk after every
        iteration. The dumped SD-Tree can be visualized with the accompanying
        visualizer tool.
        Default = false
    */
    bool m_dumpSDTree;

    /// The time at which rendering started.
    std::chrono::steady_clock::time_point m_startTime;
    ref<Film> m_film;

public:
    MTS_DECLARE_CLASS()

    static std::vector<RawImageSample> s_amisBuffer;
    static std::map<int, std::priority_queue<RawImageSample>> s_amisBufferThreadlocal;
    mutable ref<ImageBlock> m_amisImage;
};

std::vector<RawImageSample> GuidedPathTracerAMISPathspace::s_amisBuffer;
std::map<int, std::priority_queue<RawImageSample>> GuidedPathTracerAMISPathspace::s_amisBufferThreadlocal;

MTS_IMPLEMENT_CLASS(GuidedPathTracerAMISPathspace, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(GuidedPathTracerAMISPathspace, "Guided path tracer AMIS Experimental");
MTS_NAMESPACE_END
