#pragma once

#include "typedefs.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include <string>

struct DAGInfo;

struct HashDAGFactory
{
	static void load_from_DAG(
		HashDAG& outDag,
		const BasicDAG& inDag,
		uint32 numPages);

	static void load_colors_from_DAG(
		HashDAGColors& outDagColors,
		const BasicDAG& inDag,
		const BasicDAGCompressedColors& inDagColors);

	static void save_dag_to_file(const DAGInfo& info, const HashDAG& dag, const std::string& path);
	static void load_dag_from_file(DAGInfo& info, HashDAG& dag, const std::string& path);
};