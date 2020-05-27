#pragma once

#include "path.h"
#include "typedefs.h"

#include <x86intrin.h>

class PathAVX
{
	public:
		explicit PathAVX( uint32 aX, uint32 aY, uint32 aZ  )
			: x( _mm256_set1_epi32( aX ) )
			, y( _mm256_set1_epi32( aY ) )
			, z( _mm256_set1_epi32( aZ ) )
		{}
		explicit PathAVX( Path const& aPath )
			: PathAVX( aPath.path.x, aPath.path.y, aPath.path.z )
		{}
	
	public:
		void ascend( uint32 aLevels )
		{
			x = _mm256_srli_epi32( x, aLevels );
			y = _mm256_srli_epi32( y, aLevels );
			z = _mm256_srli_epi32( z, aLevels );
		}
		void descend_to_all()
		{
			// XXX-note: Make sure this isn't reversed when w.r.t. to the
			// child order later on.
			auto const xsel = _mm256_set_epi32( 0, 0, 0, 0, 1, 1, 1, 1 );
			x = _mm256_slli_epi32( x, 1 );
			x = _mm256_or_si256( x, xsel );
			
			auto const ysel = _mm256_set_epi32( 0, 0, 1, 1, 0, 0, 1, 1 );
			y = _mm256_slli_epi32( y, 1 );
			y = _mm256_or_si256( y, ysel );

			auto const zsel = _mm256_set_epi32( 0, 1, 0, 1, 0, 1, 0, 1 );
			z = _mm256_slli_epi32( z, 1 );
			z = _mm256_or_si256( z, zsel );
		}

		__m256 as_position_x() const
		{
			return _mm256_cvtepi32_ps( x );
		}
		__m256 as_position_x( uint32 aExtraShift ) const
		{
			return _mm256_cvtepi32_ps( _mm256_slli_epi32( x, aExtraShift ) );
		}

		__m256 as_position_y() const
		{
			return _mm256_cvtepi32_ps( y );
		}
		__m256 as_position_y( uint32 aExtraShift ) const
		{
			return _mm256_cvtepi32_ps( _mm256_slli_epi32( y, aExtraShift ) );
		}

		__m256 as_position_z() const
		{
			return _mm256_cvtepi32_ps( z );
		}
		__m256 as_position_z( uint32 aExtraShift ) const
		{
			return _mm256_cvtepi32_ps( _mm256_slli_epi32( z, aExtraShift ) );
		}

		// TODO: child_index()
		// TODO: is_null()
	
	public:
		__m256i x, y, z;
};
