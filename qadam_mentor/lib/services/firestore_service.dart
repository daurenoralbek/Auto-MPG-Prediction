import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:uuid/uuid.dart';
import '../config/app_constants.dart';
import '../models/mentor_model.dart';
import '../models/booking_model.dart';
import '../models/chat_model.dart';
import '../models/message_model.dart';
import '../models/user_model.dart';

class FirestoreService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  final _uuid = const Uuid();

  // ─── Users ──────────────────────────────────────────────────────────────────

  Future<UserModel?> getUser(String uid) async {
    final doc =
        await _db.collection(AppConstants.usersCollection).doc(uid).get();
    if (!doc.exists) return null;
    return UserModel.fromMap(doc.data()!);
  }

  Future<void> updateUser(String uid, Map<String, dynamic> data) async {
    await _db.collection(AppConstants.usersCollection).doc(uid).update(data);
  }

  // ─── Mentors ────────────────────────────────────────────────────────────────

  Stream<List<MentorModel>> getMentorsStream() {
    return _db
        .collection(AppConstants.mentorsCollection)
        .orderBy('rating', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => MentorModel.fromMap(d.data(), d.id))
            .toList());
  }

  Stream<List<MentorModel>> getMentorsByCategory(String categoryId) {
    return _db
        .collection(AppConstants.mentorsCollection)
        .where('categories', arrayContains: categoryId)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => MentorModel.fromMap(d.data(), d.id))
            .toList());
  }

  Future<MentorModel?> getMentor(String mentorId) async {
    final doc = await _db
        .collection(AppConstants.mentorsCollection)
        .doc(mentorId)
        .get();
    if (!doc.exists) return null;
    return MentorModel.fromMap(doc.data()!, doc.id);
  }

  Future<void> updateMentor(String mentorId, Map<String, dynamic> data) async {
    await _db
        .collection(AppConstants.mentorsCollection)
        .doc(mentorId)
        .update(data);
  }

  // ─── Bookings ───────────────────────────────────────────────────────────────

  Future<String> createBooking(BookingModel booking) async {
    final docId = _uuid.v4();
    await _db
        .collection(AppConstants.bookingsCollection)
        .doc(docId)
        .set(booking.toMap());
    return docId;
  }

  Stream<List<BookingModel>> getStudentBookings(String studentId) {
    return _db
        .collection(AppConstants.bookingsCollection)
        .where('studentId', isEqualTo: studentId)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => BookingModel.fromMap(d.data(), d.id))
            .toList());
  }

  Stream<List<BookingModel>> getMentorBookings(String mentorId) {
    return _db
        .collection(AppConstants.bookingsCollection)
        .where('mentorId', isEqualTo: mentorId)
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => BookingModel.fromMap(d.data(), d.id))
            .toList());
  }

  Future<void> updateBookingStatus(String bookingId, String status) async {
    await _db
        .collection(AppConstants.bookingsCollection)
        .doc(bookingId)
        .update({'status': status});
  }

  Future<void> deleteBooking(String bookingId) async {
    await _db
        .collection(AppConstants.bookingsCollection)
        .doc(bookingId)
        .delete();
  }

  // ─── Chats ──────────────────────────────────────────────────────────────────

  Future<String> getOrCreateChat({
    required String studentId,
    required String mentorId,
    required String studentName,
    required String mentorName,
    String? studentPhoto,
    String? mentorPhoto,
  }) async {
    final existing = await _db
        .collection(AppConstants.chatsCollection)
        .where('participants', arrayContains: studentId)
        .get();

    for (final doc in existing.docs) {
      final data = doc.data();
      final participants = List<String>.from(data['participants'] as List);
      if (participants.contains(mentorId)) {
        return doc.id;
      }
    }

    final chatId = _uuid.v4();
    await _db
        .collection(AppConstants.chatsCollection)
        .doc(chatId)
        .set({
      'participants': [studentId, mentorId],
      'studentId': studentId,
      'mentorId': mentorId,
      'studentName': studentName,
      'mentorName': mentorName,
      'studentPhoto': studentPhoto,
      'mentorPhoto': mentorPhoto,
      'lastMessage': '',
      'lastMessageTime': Timestamp.now(),
      'unreadCount': 0,
    });
    return chatId;
  }

  Stream<List<ChatModel>> getUserChats(String uid) {
    return _db
        .collection(AppConstants.chatsCollection)
        .where('participants', arrayContains: uid)
        .orderBy('lastMessageTime', descending: true)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => ChatModel.fromMap(d.data(), d.id))
            .toList());
  }

  Stream<List<MessageModel>> getMessages(String chatId) {
    return _db
        .collection(AppConstants.chatsCollection)
        .doc(chatId)
        .collection(AppConstants.messagesCollection)
        .orderBy('timestamp', descending: false)
        .snapshots()
        .map((snap) => snap.docs
            .map((d) => MessageModel.fromMap(d.data(), d.id))
            .toList());
  }

  Future<void> sendMessage({
    required String chatId,
    required String senderId,
    required String text,
  }) async {
    final msgId = _uuid.v4();
    final now = Timestamp.now();
    await _db
        .collection(AppConstants.chatsCollection)
        .doc(chatId)
        .collection(AppConstants.messagesCollection)
        .doc(msgId)
        .set({
      'senderId': senderId,
      'text': text,
      'timestamp': now,
      'isRead': false,
    });
    await _db
        .collection(AppConstants.chatsCollection)
        .doc(chatId)
        .update({
      'lastMessage': text,
      'lastMessageTime': now,
    });
  }

  // ─── Favorites ──────────────────────────────────────────────────────────────

  Future<void> addFavorite(String userId, String mentorId) async {
    await _db
        .collection(AppConstants.favoritesCollection)
        .doc('${userId}_$mentorId')
        .set({
      'userId': userId,
      'mentorId': mentorId,
      'createdAt': Timestamp.now(),
    });
  }

  Future<void> removeFavorite(String userId, String mentorId) async {
    await _db
        .collection(AppConstants.favoritesCollection)
        .doc('${userId}_$mentorId')
        .delete();
  }

  Stream<List<String>> getFavoriteMentorIds(String userId) {
    return _db
        .collection(AppConstants.favoritesCollection)
        .where('userId', isEqualTo: userId)
        .snapshots()
        .map((snap) =>
            snap.docs.map((d) => d.data()['mentorId'] as String).toList());
  }

  Future<bool> isFavorite(String userId, String mentorId) async {
    final doc = await _db
        .collection(AppConstants.favoritesCollection)
        .doc('${userId}_$mentorId')
        .get();
    return doc.exists;
  }
}
