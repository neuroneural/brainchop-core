/* Exercise the real orchestration with injected backend results. No network
   arithmetic is mocked in the numerical regression tests. */
#define main brainchopc_cli_main
#define bc_cpu_backend bc_test_cpu_backend
#include "../src/brainchopc.c"
#undef bc_cpu_backend
#undef main
#include <assert.h>

static int failure_mode, cpu_calls;

static int healthy(const unsigned char *input, float *out, const float *weights,
		const bc_inference_options *options) {
	(void)input; (void)weights;
	cpu_calls++;
	assert(!options->tissue_priors_prepared || !*options->tissue_priors_prepared);
	memset(out, 0, 256u * 256u * 256u * sizeof(*out));
	for (size_t z=64; z<192; z++) for (size_t y=64; y<192; y++)
	for (size_t x=64; x<192; x++) out[x+y*256+z*65536]=1;
	return 0;
}

const bc_backend bc_test_cpu_backend = {.name="test-cpu", .segment=healthy};

static int broken(const unsigned char *input, float *out, const float *weights,
		const bc_inference_options *options) {
	healthy(input, out, weights, options);
	cpu_calls--;
	const size_t n=256u*256u*256u;
	switch (failure_mode) {
	case 0: out[0]=NAN; break;
	case 1: memset(out,0,n*sizeof(*out)); break;
	case 2: for(size_t i=0;i<n;i++)out[i]=1; break;
	case 3:
		for(size_t i=0;i<n;i++)out[i]=i<n/2 ? 1+(i+i/256+i/65536)%2 : 0;
		break;
	case 4: out[0]=18; break;
	case 5: out[0]=1.5f; break;
	default: return 1;
	}
	return 0; /* A backend reporting success must not bypass output checks. */
}

int main(int argc, char **argv) {
	assert(argc==2);
	nifti_image *input=nifti_image_read(argv[1],1);
	assert(input && !brainchopc_convert(input,DT_FLOAT32));
	bc_cli_options cli={0}; cli.save_conform=1; cli.adapter=-1;
	const bc_backend bad={.name="injected-failure", .segment=broken};
	float weights=0;
	for(failure_mode=0;failure_mode<7;failure_mode++) {
		nifti_image *output=(nifti_image *)(uintptr_t)1;
		cpu_calls=0;
		assert(run_model(input,DT_FLOAT32,0,&weights,&bad,0,&cli,&output));
		assert(!output && !cpu_calls);
		assert(!run_model(input,DT_FLOAT32,0,&weights,&bad,1,&cli,&output));
		assert(output && cpu_calls==1);
		assert(strstr(output->descrip, "test-cpu") != NULL);
		nifti_image_free(output);
	}
	nifti_image_free(input);
	puts("injected categorical failures: explicit refusal and automatic CPU fallback PASS");
	return 0;
}
