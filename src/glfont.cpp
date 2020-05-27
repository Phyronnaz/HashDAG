#ifdef _MSC_VER
#pragma warning ( disable : 4820 4365 4668 5039 4710 4702 4711 )
#endif

#include "glfont.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <algorithm>

#include <fontstash.h>

#include <GL/glew.h>

#include "typedefs.h"

#define GL_CHECKPOINT_ALWAYS_()                                              \
	for( auto err = glGetError(); GL_NO_ERROR != err; err = GL_NO_ERROR ) {  \
		std::fprintf( stderr, "%s:%d: %d\n", __FILE__, __LINE__, err );      \
		std::abort();                                                        \
	} /*ENDM*/

#if ENABLE_CHECKS
#	define GL_CHECKPOINT_DEBUG_() GL_CHECKPOINT_ALWAYS_()
#else
#	define GL_CHECKPOINT_DEBUG_() do {} while(0)
#endif // ~ ENABLE_CHECK

namespace
{
	enum 
	{
		eBOPos_, eBOTex_, eBOCol_,
		eBOCount_
	};
	enum
	{
		// WARNING: these must match the definitions in the shader sources!
		
		// Attributes
		eGlslPos = 0, eGlslTex = 1, eGlslCol = 2,

		// Textures
		eGlslUTex = 0,

		// Uniforms
		eGlslUSize = 0
	};

	constexpr std::size_t kBOAlign_ = 1024; // bytes
	constexpr std::size_t kInitialBOCapacity_ = 1024; // elements
	constexpr std::size_t kElementBytes_[eBOCount_] = {
		2*sizeof(float),
		2*sizeof(std::uint16_t),
		sizeof(unsigned int)
	};

	struct RenderData_
	{
		GLuint tex = 0;
		std::size_t width, height;

		GLuint prog = 0;

		glf::Buffer* activeBuffer = nullptr;
	};

	void resize_gpu_( glf::Buffer* );

	void load_prog_( void* );
	GLuint load_shader_( char const*, std::size_t, GLenum aType );
}

namespace glf
{
	struct Context
	{
		FONScontext* fons;
		RenderData_* rdata;
	};

	struct Buffer
	{
		std::vector<float> pos;
		std::vector<std::uint16_t> tex;
		std::vector<std::uint32_t> col;

		GLuint bo = 0;
		GLuint vao = 0;
		std::size_t size, capacity, bytes;
		std::size_t offsets[eBOCount_];
	};
}

namespace
{
	int cb_render_resize_( void*, int, int );
	int cb_render_create_( void* aUserPtr, int aWidth, int aHeight )
	{
		load_prog_( aUserPtr );
		return cb_render_resize_( aUserPtr, aWidth, aHeight );
	}
	int cb_render_resize_( void* aUserPtr, int aWidth, int aHeight )
	{
		auto* rd = reinterpret_cast<RenderData_*>(aUserPtr);
		check( rd );

		GL_CHECKPOINT_ALWAYS_();
	
		if( 0 != rd->tex )
			glDeleteTextures( 1, &rd->tex );

		glCreateTextures( GL_TEXTURE_2D, 1, &rd->tex );
		glTextureStorage2D( rd->tex, 1, GL_R8, aWidth, aHeight );
		glTextureParameteri( rd->tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTextureParameteri( rd->tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

		GL_CHECKPOINT_ALWAYS_();

		rd->width   = std::size_t(aWidth);
		rd->height  = std::size_t(aHeight);

		return 1;
	}
	void cb_render_update_( void* aUserPtr, int* aRect, unsigned char const* aData )
	{
		auto* rd = reinterpret_cast<RenderData_*>(aUserPtr);
		check( rd );
		
		check( aRect );
		auto const w = aRect[2]-aRect[0];
		auto const h = aRect[3]-aRect[1];

		GL_CHECKPOINT_ALWAYS_();

		glPixelStorei( GL_UNPACK_ALIGNMENT, 1 );
		glPixelStorei( GL_UNPACK_ROW_LENGTH, GLint(rd->width) );
		glPixelStorei( GL_UNPACK_SKIP_PIXELS, aRect[0] );
		glPixelStorei( GL_UNPACK_SKIP_ROWS, aRect[1] );

		check( aData );
		glTextureSubImage2D( rd->tex, 0, aRect[0], aRect[1], w, h, GL_RED, GL_UNSIGNED_BYTE, aData );

		GL_CHECKPOINT_ALWAYS_();
	}
	void cb_render_render_( void* aUserPtr, float const* aPositions, float const* aTexCoords, unsigned int const* aColors, int aCount )
	{
		auto* rd = reinterpret_cast<RenderData_*>(aUserPtr);
		check( rd );

		auto* buff = rd->activeBuffer;
		check( buff );

		buff->pos.insert( buff->pos.end(), aPositions, aPositions + aCount*2 );
		buff->col.insert( buff->col.end(), aColors, aColors + aCount );
		for( int i = 0; i < aCount*2; i += 2 )
		{
			buff->tex.emplace_back( std::uint16_t(aTexCoords[i+0]*float(rd->width)+0.5f) );
			buff->tex.emplace_back( std::uint16_t(aTexCoords[i+1]*float(rd->height)+0.5f) );
		}
	}
	void cb_render_delete_( void* aUserPtr )
	{
		if( auto* rd = reinterpret_cast<RenderData_*>(aUserPtr) )
			delete rd;
	}


	void cb_stash_error_( void* aUserPtr, int aError, int aVal )
	{
		auto* fons = reinterpret_cast<FONScontext*>(aUserPtr);
		check( fons );

		switch( aError )
		{
			case FONS_ATLAS_FULL: {
				int w, h;
				fonsGetAtlasSize( fons, &w, &h );
				if( w < h )
					w *= 2;
				else
					h *= 2;
				fonsExpandAtlas( fons, w, h );
			} break;

			default: {
				std::fprintf( stderr, "**ERROR** unhandled error in fontstash\n" );
				std::fprintf( stderr, "**ERROR**   code = 0x%x (%d)\n", aError, aVal );
				std::fprintf( stderr, "**ERROR** bye!\n" );
				std::abort();
			} break;
		}
	}
}

namespace glf
{
	Context* make_context( char const* aFontFile, std::size_t aWidth, std::size_t aHeight, char const* aFallback )
	{
		auto* rdata = new RenderData_{};
		
		FONSparams params{};
		params.width         = 256;
		params.height        = 128;
		params.flags         = FONS_ZERO_BOTTOMLEFT;
		params.renderCreate  = &cb_render_create_;
		params.renderResize  = &cb_render_resize_;
		params.renderUpdate  = &cb_render_update_;
		params.renderDraw    = &cb_render_render_;
		params.renderDelete  = &cb_render_delete_;
		params.userPtr       = rdata;

		auto* fons = fonsCreateInternal( &params );
		check( fons );

		fonsSetErrorCallback( fons, cb_stash_error_, fons );
		
		auto const normal = fonsAddFont( fons, "normal", aFontFile );
		if( FONS_INVALID == normal )
		{
			fonsDeleteInternal( fons );
			return nullptr;
		}

		if( aFallback )
		{
			auto const fallb = fonsAddFont( fons, "normal-fallback", aFallback );
			if( FONS_INVALID != fallb )
			{
				fonsAddFallbackFont( fons, normal, fallb );
			}
			else
			{
				std::fprintf( stderr, "WARNING: fallback font '%s' not loaded!\n", aFallback );
			}
		}

		fonsSetFont( fons, normal );
		
		Context* ret = new Context{};
		ret->fons = fons;
		ret->rdata = rdata;

		resize_target( ret, aWidth, aHeight );

		return ret;
	}

	void destroy_context( Context* aContext )
	{
		check( aContext );
		fonsDeleteInternal( aContext->fons );

		delete aContext;
	}

	void resize_target( Context* aContext, std::size_t aWidth, std::size_t aHeight )
	{
		check( aContext );
		auto* rd = aContext->rdata;
		check( rd );

		GL_CHECKPOINT_ALWAYS_();
		glProgramUniform2f( rd->prog, eGlslUSize, float(aWidth), float(aHeight) );
		GL_CHECKPOINT_ALWAYS_();
	}
}

namespace glf
{
	void add_line( Context* aContext, Buffer* aBuffer, float aX, float aY, std::string_view aText )
	{
		add_line( aContext, aBuffer, EFmt::normal, aX, aY, aText );
	}
	void add_line( Context* aContext, Buffer* aBuffer, EFmt aFmt, float aX, float aY, std::string_view aText )
	{
		check( aContext && aBuffer );

		check( aContext->rdata );
		aContext->rdata->activeBuffer = aBuffer;

		auto* fons = aContext->fons;
		check( fons );

		fonsPushState( fons );
		if( !!(EFmt::embiggen & aFmt) )
			fonsSetSize( fons, 32.f );
		else
			fonsSetSize( fons, 24.f );

		if( !!(EFmt::blur & aFmt) )
		{
			fonsSetColor( fons, 0xff000000 );
			fonsSetBlur( fons, 2.f );
			fonsDrawText( fons, aX, aY, aText.data(), aText.data()+aText.size() );
		}

		if( !!(EFmt::normal & aFmt) )
		{
			fonsSetColor( fons, 0xfffbfbfb );
			fonsSetBlur( fons, 0.f );
			fonsDrawText( fons, aX, aY, aText.data(), aText.data()+aText.size() );
		}

		fonsPopState( fons );

		aContext->rdata->activeBuffer = nullptr;
	}

	void add_debug( Context* aContext, Buffer* aBuffer )
	{
		check( aContext && aBuffer );

		check( aContext->rdata );
		aContext->rdata->activeBuffer = aBuffer;

		auto* fons = aContext->fons;
		check( fons );

		fonsDrawDebug( fons, 10.f, 10.f );

		aContext->rdata->activeBuffer = nullptr;
	}
}


namespace glf
{
	Buffer* make_buffer()
	{
		return new Buffer{};
	}
	void destroy_buffer( Buffer* aBuffer )
	{
		check( aBuffer );
		delete aBuffer;
	}

	void clear_buffer( Buffer* aBuffer )
	{
		check( aBuffer );
		aBuffer->pos.clear();
		aBuffer->tex.clear();
		aBuffer->col.clear();
	}

	void draw_buffer( Context* aContext, Buffer* aBuffer )
	{
		check( aBuffer );

		auto const count = aBuffer->pos.size()/2;
		check( 2*count == aBuffer->pos.size() );
		check( 2*count == aBuffer->tex.size() );
		check( 1*count == aBuffer->col.size() );

		if( 0 == count )
			return;

		// Update GPU buffers
		GL_CHECKPOINT_DEBUG_();

		if( aBuffer->capacity < count )
			resize_gpu_( aBuffer );
			
		auto* vptr = reinterpret_cast<std::uint8_t*>(glMapNamedBufferRange( 
			aBuffer->bo, 
			0, aBuffer->bytes,
			GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT
		));
		check( vptr );

		void const* srcs[eBOCount_] = {
			aBuffer->pos.data(), aBuffer->tex.data(), aBuffer->col.data()
		};

		for( std::size_t i = 0; i < eBOCount_; ++i )
			std::memcpy( vptr+aBuffer->offsets[i], srcs[i], kElementBytes_[i]*count );

		glUnmapNamedBuffer( aBuffer->bo );

		GL_CHECKPOINT_DEBUG_();

		// Submit draw commands
		auto const* rd = aContext->rdata;
		check( rd );

		glUseProgram( rd->prog );
		glBindVertexArray( aBuffer->vao );
		glBindTextureUnit( 0, rd->tex );
		glDrawArrays( GL_TRIANGLES, 0, GLsizei(count) );
		glBindVertexArray( 0 );
		glUseProgram( 0 );

		GL_CHECKPOINT_DEBUG_();
	}
}

namespace
{
	void resize_gpu_( glf::Buffer* aBuffer )
	{
		check( aBuffer );
		
		auto const count = std::max( aBuffer->pos.size()/2, kInitialBOCapacity_ );

		GL_CHECKPOINT_ALWAYS_();

		if( aBuffer->bo )
			glDeleteBuffers( 1, &aBuffer->bo );
	
		// Setup BO
		std::size_t total = 0;
		std::size_t actual = ~std::size_t(0);
		for( int i = 0; i < int(eBOCount_); ++i )
		{
			aBuffer->offsets[i] = total;
			
			auto const size = kElementBytes_[i] * count;
			auto const aligned = (size + kBOAlign_ - 1) / kBOAlign_ * kBOAlign_;

			total += aligned;
			actual = std::min( actual, aligned / kElementBytes_[i] );
		}

		glCreateBuffers( 1, &aBuffer->bo );
		glNamedBufferStorage( aBuffer->bo, total, nullptr, GL_MAP_WRITE_BIT );

		aBuffer->size      = 0;
		aBuffer->capacity  = actual;
		aBuffer->bytes     = total;

		// Setup VAO
		if( 0 == aBuffer->vao )
			glCreateVertexArrays( 1, &aBuffer->vao );

		glBindVertexArray( aBuffer->vao );

		auto const* zptr = reinterpret_cast<std::uint8_t const*>(0);
		glBindBuffer( GL_ARRAY_BUFFER, aBuffer->bo );

		glEnableVertexAttribArray( eGlslPos );
		glVertexAttribPointer( eGlslPos, 2, GL_FLOAT, GL_FALSE, 0, zptr + aBuffer->offsets[eBOPos_] );

		glEnableVertexAttribArray( eGlslTex );
		glVertexAttribIPointer( eGlslTex, 2, GL_UNSIGNED_SHORT, 0, zptr + aBuffer->offsets[eBOTex_] );

		glEnableVertexAttribArray( eGlslCol );
		glVertexAttribPointer( eGlslCol, 4, GL_UNSIGNED_BYTE, GL_TRUE, 0, zptr + aBuffer->offsets[eBOCol_] );

		glBindVertexArray( 0 );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );

		GL_CHECKPOINT_ALWAYS_();
	}
}

namespace
{
	constexpr char const kVertexProg[] = R"GLSL(
		#version 460

		layout( location = 0 ) in vec2 iPos;
		layout( location = 1 ) in ivec2 iTex;
		layout( location = 2 ) in vec4 iCol;

		layout( location = 0 ) uniform vec2 uScreenSize;

		out vec2 v2fTex;
		out vec4 v2fCol;

		void main()
		{
			v2fTex = vec2(iTex);
			v2fCol = iCol;

			gl_Position = vec4( 2.0 * iPos / uScreenSize - vec2(1.0), 0.0, 1.0 );
		}
	)GLSL";
	constexpr char const kFragmentProg[] = R"GLSL(
		#version 460

		in vec2 v2fTex;
		in vec4 v2fCol;

		layout( binding = 0 ) uniform sampler2D uTex;

		layout( location = 0 ) out vec4 oColor;
		
		void main()
		{
			oColor = vec4(
				pow(v2fCol.rgb,vec3(1/2.2)), // ad hoc.
				texture( uTex, v2fTex / vec2(textureSize(uTex,0).xy) ).r
			);

			/*oColor = vec4(
				texture( uTex, v2fTex / vec2(textureSize(uTex,0).xy) ).rrr,
				1.0
			);*/

		}
	)GLSL";

	char const* type_to_str_( GLenum aType )
	{
		switch( aType )
		{
			case GL_VERTEX_SHADER: return "vertex";
			case GL_FRAGMENT_SHADER: return "fragment";
		}

		return "unknown";
	}
	GLuint load_shader_( char const* aSource, std::size_t aLength, GLenum aType )
	{
		GLchar const* const strings[] = { reinterpret_cast<GLchar const*>(aSource) };
		GLint const lengths[] = { GLint(aLength) };

		GL_CHECKPOINT_ALWAYS_();
		GLuint shader = glCreateShader( aType );
		glShaderSource( shader, sizeof(strings)/sizeof(strings[0]), strings, lengths );
		glCompileShader( shader );

		GLint success = 0;
		glGetShaderiv( shader, GL_COMPILE_STATUS, &success );
		if( !success )
		{
			GLint len = 0;
			glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &len );

			std::vector<GLchar> log(len);
			glGetShaderInfoLog( shader, len, &len, log.data() );

			std::fprintf( stderr, "Shader<fonstash:%s> failed: %s", type_to_str_(aType), reinterpret_cast<char const*>(log.data()) );

			glDeleteShader( shader );
			return 0;
		}
		GL_CHECKPOINT_ALWAYS_();

		return shader;
	}
	void load_prog_( void* aUserPtr )
	{
		// Load shaders
		auto const vert = load_shader_( kVertexProg, sizeof(kVertexProg), GL_VERTEX_SHADER );
		check( 0 != vert );
		
		auto const frag = load_shader_( kFragmentProg, sizeof(kFragmentProg), GL_FRAGMENT_SHADER );
		check( 0 != frag );
		
		// Create shader program
		GL_CHECKPOINT_ALWAYS_();
		GLuint prog = glCreateProgram();
		glAttachShader( prog, vert );
		glAttachShader( prog, frag );
		glLinkProgram( prog );

		glDeleteShader( vert );
		glDeleteShader( frag );

		GLint success = 0;
		glGetProgramiv( prog, GL_LINK_STATUS, &success );
		if( !success )
		{
			GLint len;
			glGetProgramiv( prog, GL_INFO_LOG_LENGTH, &len );

			std::vector<GLchar> log(len);
			glGetProgramInfoLog( prog, len, &len, log.data() );

			std::fprintf( stderr, "Program<fontstash> failed: %s", reinterpret_cast<char const*>(log.data()) );
			
			glDeleteProgram( prog );
			check( false );
		}

		// Use program
		auto* rd = reinterpret_cast<RenderData_*>(aUserPtr);
		check( rd );

		if( rd->prog )
			glDeleteProgram( rd->prog );

		GL_CHECKPOINT_ALWAYS_();
		rd->prog = prog;
	}
}

