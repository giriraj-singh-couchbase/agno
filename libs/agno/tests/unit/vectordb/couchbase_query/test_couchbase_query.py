from unittest.mock import AsyncMock, Mock, patch

import pytest
from couchbase.auth import PasswordAuthenticator
from couchbase.bucket import Bucket
from couchbase.cluster import Cluster
from couchbase.collection import Collection
from couchbase.options import ClusterOptions

from agno.knowledge.document import Document
from agno.vectordb.couchbase.couchbase import (
    CouchbaseQuery,
    OpenAIEmbedder,
    QueryVectorSearchSimilarity,
    QueryVectorSearchType,
)


# -------------------------------------------------
# Fixtures (duplicated for isolation)
# -------------------------------------------------

@pytest.fixture
def mock_cluster():
    with patch("agno.vectordb.couchbase.couchbase.Cluster") as mock_cluster:
        cluster = Mock(spec=Cluster)
        cluster.wait_until_ready.return_value = None
        mock_cluster.return_value = cluster
        yield cluster


@pytest.fixture
def mock_bucket(mock_cluster):
    bucket = Mock(spec=Bucket)
    mock_cluster.bucket.return_value = bucket
    collections_manager = Mock()
    bucket.collections.return_value = collections_manager
    mock_scope = Mock()
    mock_scope.name = "test_scope"
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    mock_scope.collections = [mock_collection]
    collections_manager.get_all_scopes.return_value = [mock_scope]
    return bucket


@pytest.fixture
def mock_collection(mock_bucket):
    collection = Mock(spec=Collection)
    return collection


@pytest.fixture
def mock_embedder():
    with patch("agno.vectordb.couchbase.couchbase.OpenAIEmbedder") as mock_embedder:
        openai_embedder = Mock(spec=OpenAIEmbedder)
        openai_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedder.return_value = openai_embedder
        return mock_embedder.return_value


@pytest.fixture
def couchbase_query_ann(mock_collection, mock_embedder):
    return CouchbaseQuery(
        bucket_name="test_bucket",
        scope_name="test_scope",
        collection_name="test_collection",
        couchbase_connection_string="couchbase://localhost",
        search_type=QueryVectorSearchType.ANN,
        similarity=QueryVectorSearchSimilarity.COSINE,
        n_probes=10,
        cluster_options=ClusterOptions(authenticator=PasswordAuthenticator("username", "password")),
        embedder=mock_embedder,
        embedding_key="embedding",
    )


@pytest.fixture
def couchbase_query_knn(mock_collection, mock_embedder):
    return CouchbaseQuery(
        bucket_name="test_bucket",
        scope_name="test_scope",
        collection_name="test_collection",
        couchbase_connection_string="couchbase://localhost",
        search_type="KNN",  # string form
        similarity="DOT",    # string form
        n_probes=None,
        cluster_options=ClusterOptions(authenticator=PasswordAuthenticator("username", "password")),
        embedder=mock_embedder,
    )


# -------------------------------------------------
# CouchbaseQuery Tests
# -------------------------------------------------

def test_couchbase_query_init_with_enums(couchbase_query_ann):
    assert couchbase_query_ann.bucket_name == "test_bucket"
    assert couchbase_query_ann.scope_name == "test_scope"
    assert couchbase_query_ann.collection_name == "test_collection"
    assert couchbase_query_ann._search_type == QueryVectorSearchType.ANN
    assert couchbase_query_ann._similarity == "COSINE"
    assert couchbase_query_ann._nprobes == 10
    assert couchbase_query_ann._embedding_key == "embedding"


def test_couchbase_query_init_with_strings(couchbase_query_knn):
    assert couchbase_query_knn._search_type == QueryVectorSearchType.KNN
    assert couchbase_query_knn._similarity == "DOT"
    assert couchbase_query_knn._nprobes is None
    assert couchbase_query_knn._embedding_key == "embedding"


def test_couchbase_query_create(couchbase_query_ann, mock_bucket):
    collections_manager = Mock()
    mock_bucket.collections.return_value = collections_manager
    couchbase_query_ann.create()
    collections_manager.create_scope.assert_called_once_with(scope_name="test_scope")


def test_couchbase_query_search_ann(couchbase_query_ann, mock_cluster, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_result = Mock()
    mock_row = Mock()
    mock_row.get.side_effect = lambda key, default=None: {
        "id": "test_id_1",
        "name": "test doc 1",
        "content": "test content 1",
        "meta_data": {"category": "test"},
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        "content_id": "content_123",
    }.get(key, default)
    mock_result.rows.return_value = [mock_row]
    mock_cluster.query.return_value = mock_result
    results = couchbase_query_ann.search("test query", limit=5)
    mock_embedder.get_embedding.assert_called_once_with("test query")
    sql = mock_cluster.query.call_args[0][0]
    assert "APPROX_VECTOR_DISTANCE" in sql and "COSINE" in sql and ", 10" in sql and "LIMIT 5" in sql
    assert len(results) == 1 and results[0].id == "test_id_1"


def test_couchbase_query_search_knn(couchbase_query_knn, mock_cluster, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_result = Mock()
    mock_row = Mock()
    mock_row.get.side_effect = lambda key, default=None: {
        "id": "test_id_2",
        "name": "test doc 2",
        "content": "test content 2",
        "meta_data": {"priority": "high"},
        "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
        "content_id": "content_456",
    }.get(key, default)
    mock_result.rows.return_value = [mock_row]
    mock_cluster.query.return_value = mock_result
    results = couchbase_query_knn.search("test query", limit=3)
    sql = mock_cluster.query.call_args[0][0]
    assert "VECTOR_DISTANCE" in sql and "DOT" in sql and ", 10" not in sql and "LIMIT 3" in sql
    assert len(results) == 1 and results[0].id == "test_id_2"


def test_couchbase_query_search_no_embedding(couchbase_query_ann, mock_embedder):
    mock_embedder.get_embedding.return_value = None
    assert couchbase_query_ann.search("test query", limit=5) == []


def test_couchbase_query_search_exception(couchbase_query_ann, mock_cluster, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_cluster.query.side_effect = Exception("Query execution failed")
    with pytest.raises(Exception, match="Query execution failed"):
        couchbase_query_ann.search("test query", limit=5)


def test_couchbase_query_search_row_processing_error(couchbase_query_ann, mock_cluster, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_result = Mock()
    mock_good_row = Mock()
    mock_good_row.get.side_effect = lambda key, default=None: {
        "id": "test_id_good",
        "name": "good doc",
        "content": "good content",
        "meta_data": {},
        "embedding": [0.1, 0.2, 0.3],
        "content_id": None,
    }.get(key, default)
    mock_result.rows.return_value = [mock_good_row]
    mock_cluster.query.return_value = mock_result
    results = couchbase_query_ann.search("test query", limit=5)
    assert len(results) == 1 and results[0].id == "test_id_good"


@pytest.mark.asyncio
async def test_couchbase_query_async_search_ann(couchbase_query_ann, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_async_cluster = AsyncMock()
    mock_result = Mock()
    async def mock_rows():
        mock_row = Mock()
        mock_row.get.side_effect = lambda key, default=None: {
            "id": "async_test_id",
            "name": "async test doc",
            "content": "async test content",
            "meta_data": {"type": "async"},
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "content_id": "async_content_123",
        }.get(key, default)
        yield mock_row
    mock_result.rows = mock_rows
    mock_async_cluster.query.return_value = mock_result
    with patch.object(couchbase_query_ann, 'get_async_cluster', return_value=mock_async_cluster):
        results = await couchbase_query_ann.async_search("async test query", limit=5)
    assert len(results) == 1 and results[0].id == "async_test_id"


@pytest.mark.asyncio
async def test_couchbase_query_async_search_with_async_embedding(couchbase_query_ann, mock_embedder):
    mock_embedder.async_get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    mock_async_cluster = AsyncMock()
    mock_result = Mock()
    async def mock_rows():
        mock_row = Mock()
        mock_row.get.side_effect = lambda key, default=None: {
            "id": "async_embed_test_id",
            "name": "async embed test doc",
            "content": "async embed test content",
            "meta_data": {},
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
            "content_id": "async_embed_content_123",
        }.get(key, default)
        yield mock_row
    mock_result.rows = mock_rows
    mock_async_cluster.query.return_value = mock_result
    with patch.object(couchbase_query_ann, 'get_async_cluster', return_value=mock_async_cluster):
        results = await couchbase_query_ann.async_search("async embed test query", limit=3)
    mock_embedder.async_get_embedding.assert_called_once_with("async embed test query")
    assert len(results) == 1 and results[0].id == "async_embed_test_id"


@pytest.mark.asyncio
async def test_couchbase_query_async_search_knn(couchbase_query_knn, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.5, 0.4, 0.3, 0.2, 0.1]
    mock_async_cluster = AsyncMock()
    mock_result = Mock()
    async def mock_rows():
        mock_row = Mock()
        mock_row.get.side_effect = lambda key, default=None: {
            "id": "knn_async_test_id",
            "name": "knn async test doc",
            "content": "knn async test content",
            "meta_data": {"search_type": "knn"},
            "embedding": [0.5, 0.4, 0.3, 0.2, 0.1],
            "content_id": "knn_async_content_123",
        }.get(key, default)
        yield mock_row
    mock_result.rows = mock_rows
    mock_async_cluster.query.return_value = mock_result
    with patch.object(couchbase_query_knn, 'get_async_cluster', return_value=mock_async_cluster):
        results = await couchbase_query_knn.async_search("knn async test query", limit=2)
    assert len(results) == 1 and results[0].id == "knn_async_test_id"


@pytest.mark.asyncio
async def test_couchbase_query_async_search_no_embedding(couchbase_query_ann, mock_embedder):
    mock_embedder.get_embedding.return_value = None
    if hasattr(mock_embedder, 'async_get_embedding'):
        mock_embedder.async_get_embedding = AsyncMock(return_value=None)
    results = await couchbase_query_ann.async_search("test query", limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_couchbase_query_async_search_exception(couchbase_query_ann, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_async_cluster = AsyncMock()
    mock_async_cluster.query.side_effect = Exception("Async query execution failed")
    with patch.object(couchbase_query_ann, 'get_async_cluster', return_value=mock_async_cluster):
        with pytest.raises(Exception, match="Async query execution failed"):
            await couchbase_query_ann.async_search("test query", limit=5)


@pytest.mark.asyncio
async def test_couchbase_query_async_search_row_processing_error(couchbase_query_ann, mock_embedder):
    mock_embedder.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_async_cluster = AsyncMock()
    mock_result = Mock()
    async def mock_rows_with_error():
        bad_row = Mock(); bad_row.get.side_effect = lambda key, default=None: (_ for _ in ()).throw(KeyError("Missing async field"))
        yield bad_row
        good_row = Mock(); good_row.get.side_effect = lambda key, default=None: {
            "id": "async_good_id",
            "name": "async good doc",
            "content": "async good content",
            "meta_data": {},
            "embedding": [0.1, 0.2, 0.3],
            "content_id": None,
        }.get(key, default)
        yield good_row
    mock_result.rows = mock_rows_with_error
    mock_async_cluster.query.return_value = mock_result
    with patch.object(couchbase_query_ann, 'get_async_cluster', return_value=mock_async_cluster):
        results = await couchbase_query_ann.async_search("test query", limit=5)
    assert len(results) == 1 and results[0].id == "async_good_id"


@pytest.mark.asyncio
async def test_couchbase_query_async_create(couchbase_query_ann, mock_bucket):
    collections_manager = Mock()
    mock_bucket.collections.return_value = collections_manager
    with patch.object(couchbase_query_ann, '_async_create_collection_and_scope', new_callable=AsyncMock) as mock_async_create:
        await couchbase_query_ann.async_create()
        mock_async_create.assert_called_once()


def test_couchbase_query_enum_values():
    assert QueryVectorSearchType.ANN == "ANN"
    assert QueryVectorSearchType.KNN == "KNN"
    assert QueryVectorSearchSimilarity.COSINE == "COSINE"
    assert QueryVectorSearchSimilarity.DOT == "DOT"
    assert QueryVectorSearchSimilarity.L2 == "L2"
    assert QueryVectorSearchSimilarity.EUCLIDEAN == "EUCLIDEAN"
    assert QueryVectorSearchSimilarity.L2_SQUARED == "L2_SQUARED"
    assert QueryVectorSearchSimilarity.EUCLIDEAN_SQUARED == "EUCLIDEAN_SQUARED"


def test_couchbase_query_similarity_string_conversion(mock_collection, mock_embedder):
    query_db = CouchbaseQuery(
        bucket_name="test_bucket",
        scope_name="test_scope",
        collection_name="test_collection",
        couchbase_connection_string="couchbase://localhost",
        search_type=QueryVectorSearchType.ANN,
        similarity="cosine",
        n_probes=5,
        cluster_options=ClusterOptions(authenticator=PasswordAuthenticator("username", "password")),
        embedder=mock_embedder,
    )
    assert query_db._similarity == "COSINE"


def test_couchbase_query_custom_embedding_key(mock_collection, mock_embedder):
    query_db = CouchbaseQuery(
        bucket_name="test_bucket",
        scope_name="test_scope",
        collection_name="test_collection",
        couchbase_connection_string="couchbase://localhost",
        search_type=QueryVectorSearchType.KNN,
        similarity=QueryVectorSearchSimilarity.L2,
        n_probes=None,
        cluster_options=ClusterOptions(authenticator=PasswordAuthenticator("username", "password")),
        embedder=mock_embedder,
        embedding_key="custom_embedding_field",
    )
    assert query_db._embedding_key == "custom_embedding_field"
