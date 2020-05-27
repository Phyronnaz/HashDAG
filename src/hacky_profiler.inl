namespace detail
{
	inline 
	HackyProfiler& HackyProfiler::instance() noexcept
	{
		static HackyProfiler inst;
			return inst;
	}
}