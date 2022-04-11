import urllib.request


if __name__ == '__main__': 

    url = 'https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_rna.fna.gz'
    print('Beginning download of ', url)

    urllib.request.urlretrieve(url, '../data/GRCh38_latest_rna.fna.gz')