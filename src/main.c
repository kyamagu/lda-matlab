// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * main
 *
 */

int main(int argc, char* argv[])
{
    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)

    lda_corpus* corpus;
    lda_settings* settings = get_settings();

    long t1;
    (void) time(&t1);
    seedMT(t1);
    // seedMT(4357U);

    if (argc > 1)
    {
        if (strcmp(argv[1], "est")==0)
        {
            settings->INITIAL_ALPHA = atof(argv[2]);
            settings->NTOPICS = atoi(argv[3]);
            read_settings(argv[4]);
            corpus = read_data(argv[5]);
            make_directory(argv[7]);
            run_em(argv[6], argv[7], corpus);
        }
        if (strcmp(argv[1], "inf")==0)
        {
            read_settings(argv[2]);
            corpus = read_data(argv[4]);
            infer(argv[3], argv[5], corpus);
        }
    }
    else
    {
        printf("usage : lda est [initial alpha] [k] [settings] [data] [random/seeded/*] [directory]\n");
        printf("        lda inf [settings] [model] [data] [name]\n");
    }
    return(0);
}