"""
이 파일은 S3ParquetReader 클래스를 정의하고 있으며, 이 클래스는 AWS S3에서 Parquet 파일을 효율적으로 읽어들이기 위한 기능을 제공한다. 클래스의 주요 기능은 다음과 같다:

1. 클래스 초기화 (__init__ 메소드):
    • AWS S3 자격 증명 및 지역 설정을 받아 boto3 세션을 초기화한다.
    • 멀티스레딩 사용 여부(use_multithreading)와 최대 스레드 수(max_threads)를 설정할 수 있다. 멀티스레딩 사용 시, 파일 로딩 속도가 향상될 수 있다.
    
2. 단일 Parquet 파일 읽기 (read_parquet_from_s3 메소드):
    • 지정된 S3 버킷과 키를 사용하여 단일 Parquet 파일을 읽고, Pandas DataFrame으로 반환한다.

3. 다중 Parquet 파일 읽기 (read_multiple_parquets_from_s3 메소드):
    • 지정된 S3 경로의 모든 Parquet 파일을 찾아 이를 합쳐 하나의 DataFrame으로 반환한다.
    • 필요에 따라 경로의 특정 부분(예: 'year', 'month')을 추출하여 DataFrame의 컬럼으로 추가할 수 있다.
    • 멀티스레딩을 활성화한 경우, concurrent.futures.ThreadPoolExecutor를 사용하여 파일 로딩을 병렬 처리한다.
4. 카테고리별 Parquet 파일 읽기 (read_all_categories 메소드):
    • 여러 카테고리(접두사가 다른 경로)에 걸쳐 있는 Parquet 파일들을 읽어 합치는 기능을 제공한다.
5. 도우미 메소드들:
    • _parse_s3_key: S3 키에서 지정된 부분을 추출한다.
    • _get_s3_keys: 주어진 경로에 해당하는 S3 키 리스트를 가져온다.
    • _print_verbose: 파일 로딩 과정을 출력한다.
    • 이 클래스는 S3 상의 대량의 Parquet 데이터를 효율적으로 읽어들이고 처리하는 데 유용하며, 특히 대규모 데이터 처리 작업에서 성능 향상을 위해 멀티스레딩 옵션을 제공한다.
"""

# pip install boto3 pandas pyarrow

# import libraries
import boto3
import pandas as pd
import io
import concurrent.futures
import os

class S3ParquetReader:
    def __init__(self, s3_config: dict, use_multithreading: bool = False, max_threads: int = None):
        """
        AWS S3에서 Parquet 파일을 읽기 위한 클래스를 초기화한다.
        :param s3_config: AWS 자격 증명 및 지역 정보를 포함한 사전.
        :param use_multithreading: 멀티스레딩을 사용할지 여부.
        :param max_threads: 사용할 최대 스레드 수. None이면 기본값 사용.
        """
        session = boto3.session.Session(**s3_config)
        self.s3_client = session.client('s3')
        self.s3_resource = session.resource('s3')
        self.use_multithreading = use_multithreading
        self.max_threads = max_threads if max_threads is not None else min(32, (os.cpu_count() or 1) + 4)

    def read_parquet_from_s3(self, key: str, bucket: str, **args):
        """
        S3에서 Parquet 파일을 읽어 Pandas DataFrame으로 반환한다.
        :param key: Parquet 파일의 S3 키.
        :param bucket: 파일이 저장된 S3 버킷.
        :return: Parquet 파일의 데이터를 포함하는 Pandas DataFrame.
        """
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            with io.BytesIO(obj['Body'].read()) as f:
                return pd.read_parquet(f, **args)
        except Exception as e:
            print(f"{bucket} 버킷의 {key} 읽기 오류: {e}")
            return pd.DataFrame()

    def read_multiple_parquets_from_s3(self, filepath: str, bucket: str, extract_columns=None, verbose: bool = False, **args):
        """
        주어진 S3 경로에서 여러 Parquet 파일을 읽고 하나의 DataFrame으로 합친다. 
        필요한 경우, 경로의 특정 부분을 컬럼으로 추출한다.
        :param filepath: Parquet 파일이 저장된 S3 경로.
        :param bucket: S3 버킷.
        :param extract_columns: 추출하고 싶은 경로 부분의 리스트. 예: ['year', 'month']
        :param verbose: True인 경우, 로드되는 파일을 출력한다.
        :return: 모든 Parquet 파일을 합친 Pandas DataFrame.
        """
        s3_keys = self._get_s3_keys(filepath, bucket)
        if not s3_keys:
            print(f'{bucket}의 {filepath}에 Parquet 파일이 없습니다.')
            return pd.DataFrame()
        if verbose:
            self._print_verbose(s3_keys)

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = {executor.submit(self.read_parquet_from_s3, key, bucket=bucket, **args): key for key in s3_keys}
                dfs = []
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    try:
                        df = future.result()
                        if extract_columns:
                            path_details = self._parse_s3_key(key, extract_columns)
                            for col, val in path_details.items():
                                df[col] = val
                        dfs.append(df)
                    except Exception as e:
                        print(f"{bucket} 버킷 {key} 읽기 오류: {e}")
        else:
            dfs = [self.read_parquet_from_s3(key, bucket=bucket, **args) for key in s3_keys]
            for i, df in enumerate(dfs):
                if extract_columns:
                    path_details = self._parse_s3_key(s3_keys[i], extract_columns)
                    for col, val in path_details.items():
                        df[col] = val

        return pd.concat(dfs, ignore_index=True)


    def read_all_categories(self, category_prefixes: list, bucket: str, verbose: bool = False, **args):
        """
        여러 카테고리 접두사에 해당하는 Parquet 파일을 읽고 합친다.
        :param category_prefixes: 읽을 카테고리 접두사 리스트.
        :param bucket: S3 버킷.
        :param verbose: True인 경우, 상세한 출력을 보여준다.
        :return: 모든 카테고리의 데이터를 합친 DataFrame.
        """
        dfs = [self.read_multiple_parquets_from_s3(prefix, bucket, verbose=verbose, **args) for prefix in category_prefixes]
        return pd.concat(dfs, ignore_index=True)

    def _parse_s3_key(self, key: str, extract_columns: list):
        """
        S3 키에서 지정된 부분의 값을 추출한다.
        :param key: S3 키.
        :param extract_columns: 추출할 경로 부분의 리스트.
        :return: 추출된 부분의 값이 포함된 사전.
        """
        path_parts = key.split('/')
        details = {}
        for part in path_parts:
            if '=' in part:
                col, val = part.split('=', 1)
                if col in extract_columns:
                    details[col] = val
        return details

    def _get_s3_keys(self, filepath: str, bucket: str):
        """
        주어진 경로에 해당하는 S3 키 리스트를 가져온다.
        :param filepath: 검색할 S3 경로.
        :param bucket: S3 버킷.
        :return: 해당 경로에 맞는 S3 키의 리스트.
        """
        if not filepath.endswith('/'):
            filepath += '/'
        return [item.key for item in self.s3_resource.Bucket(bucket).objects.filter(Prefix=filepath)
                if item.key.endswith('.parquet')]

    def _print_verbose(self, s3_keys):
        print('Parquet 파일 로딩 중:')
        for p in s3_keys: 
            print(p)
            
# 사용 예:
s3_config = {
    'aws_access_key_id': 'YOUR_ACCESS_KEY_ID',
    'aws_secret_access_key': 'YOUR_SECRET_ACCESS_KEY',
    'region_name': 'YOUR_REGION'
}

bucket_name = 'your-bucket-name'
file_path = 'your/file/path/'

# 멀티스레딩을 사용하고 최대 스레드 수를 10으로 설정하여 S3ParquetReader 인스턴스 생성
reader = S3ParquetReader(s3_config, use_multithreading=True, max_threads=10)

# 여러 Parquet 파일 읽기
multiple_parquets_df = reader.read_multiple_parquets_from_s3(file_path, bucket_name, verbose=True)
print(multiple_parquets_df.head())

# 파티션을 추출하여 컬럼으로 추가
extract_columns = ['year', 'month', 'day']
multiple_parquets_df = reader.read_multiple_parquets_from_s3(file_path, bucket_name, extract_columns=extract_columns, verbose=True)
print(multiple_parquets_df.head())

# 다양한 카테고리 접두사를 사용하여 여러 Parquet 파일 읽기
category_prefixes = ['your/file/path1/', 'your/file/path2/']
all_categories_df = reader.read_all_categories(category_prefixes, bucket_name, verbose=True)
print(all_categories_df.head())