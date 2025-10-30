# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the llm-service project

import pytest

try:
    from llm_service.protocol.protocol import (
        ProfileRequest,
        ProfileResponse,
        RequestType,
        ResponseType,
    )
except ImportError:
    pytest.skip(
        "vllm dependencies not available for integration test",
        allow_module_level=True,
    )


class TestProfilingProtocol:
    """Test suite for profiling protocol classes."""

    def test_profile_request_creation(self):
        """Test creating a ProfileRequest."""
        request_id = "test-request-123"
        profile_req = ProfileRequest(request_id=request_id)
        assert profile_req.request_id == request_id

    def test_profile_response_creation_default(self):
        """Test creating a ProfileResponse with default status."""
        request_id = "test-request-456"
        profile_resp = ProfileResponse(request_id=request_id)
        assert profile_resp.request_id == request_id
        assert profile_resp.status == "OK"

    def test_profile_response_creation_custom_status(self):
        """Test creating a ProfileResponse with custom status."""
        request_id = "test-request-789"
        profile_resp = ProfileResponse(request_id=request_id, status="COMPLETED")
        assert profile_resp.request_id == request_id
        assert profile_resp.status == "COMPLETED"

    def test_start_profile_request_type(self):
        """Test START_PROFILE request type constant."""
        assert RequestType.START_PROFILE == b"\x04"

    def test_stop_profile_request_type(self):
        """Test STOP_PROFILE request type constant."""
        assert RequestType.STOP_PROFILE == b"\x05"

    def test_profile_response_type(self):
        """Test PROFILE response type constant."""
        assert ResponseType.PROFILE == b"\x04"
