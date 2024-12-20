#bash test_mem_llama7B.sh TP PP CP DP SP
# FP8_init, opt dytpe: bf16, grad dtype: bf16
bash test_mem_llama7B.sh 1	1	1	8	0 4096	1	32
bash test_mem_llama7B.sh 2	1	1	4	0 4096	1	32
bash test_mem_llama7B.sh 2	1	1	4	1 4096	1	32
bash test_mem_llama7B.sh 1	2	1	4	0 4096	1	32
bash test_mem_llama7B.sh 1	1	2	4	0 4096	1	32
bash test_mem_llama7B.sh 2	2	1	2	0 4096	1	32
bash test_mem_llama7B.sh 2	2	1	2	1 4096	1	32
bash test_mem_llama7B.sh 2	2	2	1	0 4096	1	32
bash test_mem_llama7B.sh 2	2	2	1	1 4096	1	32
bash test_mem_llama7B.sh 1	1	1	8	0 4096	2	32
bash test_mem_llama7B.sh 1	1	1	8	0 4096	1	64
bash test_mem_llama7B.sh 1	1	1	8	0 2048	1	32
