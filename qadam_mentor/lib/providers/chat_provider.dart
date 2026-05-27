import 'package:flutter/material.dart';
import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../services/firestore_service.dart';

class ChatProvider extends ChangeNotifier {
  final FirestoreService _service = FirestoreService();

  List<ChatModel> _chats = [];
  List<MessageModel> _messages = [];
  bool _isLoading = false;

  List<ChatModel> get chats => _chats;
  List<MessageModel> get messages => _messages;
  bool get isLoading => _isLoading;

  void listenToChats(String uid) {
    _service.getUserChats(uid).listen((list) {
      _chats = list;
      notifyListeners();
    });
  }

  void listenToMessages(String chatId) {
    _service.getMessages(chatId).listen((list) {
      _messages = list;
      notifyListeners();
    });
  }

  void clearMessages() {
    _messages = [];
    notifyListeners();
  }

  Future<String> getOrCreateChat({
    required String studentId,
    required String mentorId,
    required String studentName,
    required String mentorName,
    String? studentPhoto,
    String? mentorPhoto,
  }) async {
    return _service.getOrCreateChat(
      studentId: studentId,
      mentorId: mentorId,
      studentName: studentName,
      mentorName: mentorName,
      studentPhoto: studentPhoto,
      mentorPhoto: mentorPhoto,
    );
  }

  Future<void> sendMessage({
    required String chatId,
    required String senderId,
    required String text,
  }) async {
    await _service.sendMessage(
      chatId: chatId,
      senderId: senderId,
      text: text,
    );
  }
}
