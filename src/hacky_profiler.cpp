#include "hacky_profiler.hpp"

#include <chrono>

#include <cstdio>
#include <cassert>

namespace detail
{
	HackyCounter::HackyCounter( char const* aName )
		: name(aName)
	{
		HackyProfiler::instance().register_counter( this );
	}


	void HackyProfiler::register_counter( HackyCounter* aCounter )
	{
		assert( aCounter );
		mKnownCounters.emplace_back( aCounter );
	}
	void HackyProfiler::frame_advance() noexcept
	{
		std::size_t totalHits = 0;
		for( auto& counter : mKnownCounters )
			totalHits += counter->hits;

		if( 0 == totalHits )
		{
			//std::printf( "Frame %4zu: nothing\n", mFrame );
			return;
		}
		
		std::printf( "Frame %4zu: %zu counters:\n", mFrame, mKnownCounters.size() );
		using Msf_ = std::chrono::duration<float, std::milli>;
		for( auto& counter : mKnownCounters )
		{
			std::printf( "%24s: %.2fms (%zu hits)\n", counter->name, std::chrono::duration_cast<Msf_>(counter->accum).count(), counter->hits );
			counter->accum = HackyClock::duration{};
			counter->hits = 0;
		}

		++mFrame;
	}
}
