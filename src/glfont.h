/* Uses fontstash by Mikko Mononen. Largely mirrors glfontstash.h, but uses
 * a more modern OpenGL
 */

#ifndef GLFONT_H_2D16D9ED_766E_4D06_A564_CE6D84FA0D43
#define GLFONT_H_2D16D9ED_766E_4D06_A564_CE6D84FA0D43

#include <cstddef>
#include <string_view>

namespace glf
{
	struct Context;

	Context* make_context( 
		char const*, 
		std::size_t aTargetWidth, std::size_t aTargetHeight,
		char const* = nullptr
	);

	void destroy_context( Context* );

	void resize_target( 
		Context*, 
		std::size_t aTargetWidth, std::size_t aTargetHeight
	);
	

	struct Buffer;

	Buffer* make_buffer();
	void destroy_buffer( Buffer* );


	void clear_buffer( Buffer* );
	void draw_buffer( Context*, Buffer* );

	enum class EFmt : unsigned
	{
		blur = (1<<0),
		normal = (1<<1),
		embiggen = (1<<2),

		glow = (blur|normal),
		large = (glow|embiggen),

	};


	void add_line( Context*, Buffer*, float, float, std::string_view );
	void add_line( Context*, Buffer*, EFmt, float, float, std::string_view );

	void add_debug( Context*, Buffer* );
	

	// EFmt is a bitfield...
	constexpr EFmt operator~ (EFmt aX) noexcept {
		return EFmt( ~unsigned(aX) );
	}
	constexpr bool operator! (EFmt aX) noexcept {
		return !unsigned(aX);
	}
	constexpr EFmt operator| (EFmt aX, EFmt aY) noexcept {
		return EFmt( unsigned(aX) | unsigned(aY) );
	}
	constexpr EFmt operator& (EFmt aX, EFmt aY) noexcept {
		return EFmt( unsigned(aX) & unsigned(aY) );
	}
	constexpr EFmt operator^ (EFmt aX, EFmt aY) noexcept {
		return EFmt( unsigned(aX) ^ unsigned(aY) );
	}

	// Missing: |= et al., and many more
}

#endif // GLFONT_H_2D16D9ED_766E_4D06_A564_CE6D84FA0D43
