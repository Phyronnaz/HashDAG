#pragma once

#include "typedefs.h"
#include <vector>
#include <chrono>
#include <cstddef>

#define CFG_HACK_PROFILE 0

#if CFG_HACK_PROFILE
#	define HACK_PROFILE_COUNTER_DEFINE( name )                      \
	static ::detail::HackyCounter hackyProfiler##name##_{ #name }   \
	/*ENDM*/
#	define HACK_PROFILE_SCOPED(  name )                             \
	::detail::HackyScopedRegion                                     \
		HACK_PROFILER_PASTE_(hackProfilerScope,__LINE__,_) {        \
		hackyProfiler##name##_                                      \
	}; /*ENDM*/

#	define HACK_PROFILE_START( name )                               \
	hackyProfiler##name##_.beg = ::detail::HackyClock::now();       \
	/*ENDM*/
#	define HACK_PROFILE_STOP( name )                                \
	[] {                                                            \
		auto end = ::detail::HackyClock::now();                     \
		hackyProfiler##name##_.accum += end-hackyProfiler##name##_.beg;  \
		++hackyProfiler##name##_.hits;                              \
	}() /*ENDM*/
#	define HACK_PROFILE_FRAME_ADVANCE()                             \
	::detail::HackyProfiler::instance().frame_advance()             \
	/*ENDM*/

#	define HACK_PROFILER_PASTE_(a,b,c) HACK_PROFILER_PASTE0_(a,b,c)
#	define HACK_PROFILER_PASTE0_(a,b,c) a ## b ## c

#else // !CFG_HACK_PROFILE
#	define HACK_PROFILE_COUNTER_DEFINE( name ) (void)0
#	define HACK_PROFILE_SCOPED(  name ) (void)0
#	define HACK_PROFILE_START( name ) (void)0
#	define HACK_PROFILE_STOP( name ) (void)0
#	define HACK_PROFILE_FRAME_ADVANCE() (void)0
#endif // ~ CFG_HACK_PROFILE

namespace detail
{
	using HackyClock = std::chrono::high_resolution_clock;
	
	struct HackyCounter
	{
		char const* name;
		HackyClock::duration accum;
		HackyClock::time_point beg;
		std::size_t hits;

		explicit HackyCounter( char const* );

		HackyCounter( HackyCounter const& ) = delete;
		HackyCounter& operator= (HackyCounter const&) = delete;
	};
	
	class HackyProfiler
	{
		private:
			HackyProfiler() = default;

			HackyProfiler( HackyProfiler const& ) = delete;
			HackyProfiler& operator= (HackyProfiler const&) = delete;

		public:
			void register_counter( HackyCounter* );
			void frame_advance() noexcept;

		public:
			static HackyProfiler& instance() noexcept;

		private:
			std::vector<HackyCounter*> mKnownCounters;
			std::size_t mFrame = 0;
	};

	struct HackyScopedRegion
	{
		explicit HackyScopedRegion( HackyCounter& aCounter ) noexcept 
			: counter(aCounter)
		{
			beg = HackyClock::now();
		};
		~HackyScopedRegion()
		{
			auto end = ::detail::HackyClock::now();
			counter.accum += end-beg;
			++counter.hits;
		};

		HackyCounter& counter;
		HackyClock::time_point beg;
	};
}

#include "hacky_profiler.inl"