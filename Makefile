# multicollinearity_llm, a multicollinearity-based compression C program,
# identifies and removes highly correlated weights in neural networks,
# thereby reducing redundancy. The algorithm simplifies the model structure,
# potentially improving generalisation and computational efficiency whilst
# maintaining performance.
#
# Copyright (c) 2024 Finbarrs Oketunji
# Written by Finbarrs Oketunji <f@finbarrs.eu>
#
# This file is part of multicollinearity_llm.
#
# multicollinearity_llm is an open-source software: you are free to redistribute
# and/or modify it under the terms specified in version 3 of the GNU
# General Public License, as published by the Free Software Foundation.
#
# multicollinearity_llm is is made available with the hope that it will be beneficial,
# but it comes with NO WARRANTY whatsoever. This includes, but is not limited
# to, any implied warranties of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. For more detailed information, please refer to the
# GNU General Public License.
#
# You should have received a copy of the GNU General Public License
# along with multicollinearity_llm.  If not, visit <http://www.gnu.org/licenses/>.

CC = gcc
CFLAGS = -I./include -Wall -Wextra -O2
LDFLAGS = -lm
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

SRC = $(wildcard $(SRC_DIR)/*.c)

OBJ = $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

LIBRARY = $(LIB_DIR)/libmodelcompressor.a

all: $(LIBRARY) example ## Build the library and example

$(LIBRARY): $(OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

example: examples/compression_example.c $(LIBRARY) ## Build the example program
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -lmodelcompressor $(LDFLAGS) -o $@

clean: ## Clean up build artifacts
	rm -rf $(OBJ_DIR) $(LIB_DIR) example

help: ## Display help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: all example clean help
.DEFAULT_GOAL := help