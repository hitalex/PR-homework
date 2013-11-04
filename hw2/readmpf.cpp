// read mpf file

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(){
    int index=1; // writer index
    char* path = (char*)malloc(sizeof(char) * 100);
    char* outpath = (char*)malloc(sizeof(char) * 100);
    assert(path!=NULL && outpath!=NULL);
    
    for(index=241; index<=300; index++){
        sprintf (path, "/home/kqc/dataset/HWDB1.1/HWDB1.1tst/1%03d.mpf", index);
        FILE* f = fopen(path, "rb");
        
        if(f == NULL){
            printf("File not exit: %s\n", path);
            return 1;
        }
        
        printf("Reading file: %s\n", path);
        
        long int header_size;
        size_t result = fread (&header_size, 4, 1, f); // read header size
        printf("Header size: %d\n", header_size);
        
        long int llustr_size = header_size - 62;
        
        char* format_code = (char*)malloc(sizeof(char) * 8);
        char* illustration_text = (char*)malloc(sizeof(char) * llustr_size);
        char* code_type = (char*)malloc(sizeof(char) * 20);
        
        assert(format_code!=NULL && illustration_text!=NULL && code_type!=NULL)
        
        result = fread (format_code, 8, 1, f); // read format code
        result = fread (illustration_text, llustr_size, 1, f);
        result = fread (code_type, 20, 1, f);
        
        short int code_length;
        result = fread (&code_length, 2, 1, f);
        
        char* data_type = (char*)malloc(sizeof(char) * 20);
        assert(data_type!=NULL);
        result = fread (data_type, 20, 1, f);
        
        long int sample_num;
        result = fread (&sample_num, 4, 1, f);
        
        long int dim;
        result = fread (&dim, 4, 1, f);

        //printf("Label: %s\n", label);
        
        if(strcmp(data_type, "unsigned char") != 0){
            printf("Data type is not unsigned char; it is: %s\n", data_type);
            return 1;
        }
        
        unsigned char* label = (unsigned char*)malloc(sizeof(unsigned char) * code_length);
        unsigned char* vector = (unsigned char*)malloc(sizeof(unsigned char) * dim);
        assert(label!=NULL && vector!=NULL);
            
        sprintf(outpath, "/home/kqc/dataset/HWDB1.1/test/1%03d.txt", index);
        FILE* outf = fopen(outpath, "w");
        
        printf("Writing %s\n", outpath);
        int i = 0;
        for(i=0; i<sample_num; i++){
            result = fread (label, code_length, 1, f);
            if(label[0]=='F' && label[1]=='F')
                continue;
            
            result = fread (vector, sizeof(unsigned char), dim, f);
            
            fprintf(outf, "%X%X ", label[0], label[1]);
            int j=0;
            for(j=0; j<dim; j++){
                fprintf(outf, "%d ", vector[j]);
            }
            fprintf(outf, "\n");
        }
        
        fclose(f);
        fclose(outf);
    }
    
    free(path);
    free(outpath);
    
    free(format_code);
    free(illustration_text);
    free(code_type);
    free(data_type);
    free(label);
    free(vector);
    
    //printf("Code length: %d", code_length);
    return 1;
}
