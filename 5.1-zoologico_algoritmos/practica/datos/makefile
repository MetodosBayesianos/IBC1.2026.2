# ACIC 2016: covariables y los escenarios 4 (solapamiento total) y 7 (solapamiento limitado)
ESCENARIOS = 4 7

# el tarball pesa 1,2 GB: make lo borra despues de extraer
.INTERMEDIATE: ACIC2016Data.tar.gz

data_cf_all: ACIC2016Data.tar.gz
	tar -xzf $< data_cf_all/x.csv $(addprefix data_cf_all/,$(ESCENARIOS))

ACIC2016Data.tar.gz:
	curl -L https://github.com/MetodosBayesianos/IBC1.2025.1/releases/download/materiales/ACIC2016Data.tar.gz -o $@
