#define BENCHMARK 1

#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_editors.h"

void compile(HashDAG& dag, HashDAGColors& colors, HashDAGUndoRedo& undoRedo, StatsRecorder& statsRecorder, float3 center, float radius)
{
    dag.edit_threads(SphereEditor<true>(center, radius), colors, undoRedo, statsRecorder);
}