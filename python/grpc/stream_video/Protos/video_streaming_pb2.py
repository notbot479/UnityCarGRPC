# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: video_streaming.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x15video_streaming.proto\")\n\x12VideoStreamRequest\x12\x13\n\x0bvideo_chunk\x18\x01 \x01(\x0c\"&\n\x13VideoStreamResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2S\n\x15VideoStreamingService\x12:\n\x0bUploadVideo\x12\x13.VideoStreamRequest\x1a\x14.VideoStreamResponse(\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'video_streaming_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_VIDEOSTREAMREQUEST']._serialized_start=25
  _globals['_VIDEOSTREAMREQUEST']._serialized_end=66
  _globals['_VIDEOSTREAMRESPONSE']._serialized_start=68
  _globals['_VIDEOSTREAMRESPONSE']._serialized_end=106
  _globals['_VIDEOSTREAMINGSERVICE']._serialized_start=108
  _globals['_VIDEOSTREAMINGSERVICE']._serialized_end=191
# @@protoc_insertion_point(module_scope)
