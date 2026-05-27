import 'package:flutter/material.dart';
import '../models/booking_model.dart';
import '../services/firestore_service.dart';

class BookingProvider extends ChangeNotifier {
  final FirestoreService _service = FirestoreService();

  List<BookingModel> _bookings = [];
  bool _isLoading = false;
  String? _error;

  List<BookingModel> get bookings => _bookings;
  bool get isLoading => _isLoading;
  String? get error => _error;

  List<BookingModel> get pendingBookings =>
      _bookings.where((b) => b.status == 'pending').toList();
  List<BookingModel> get acceptedBookings =>
      _bookings.where((b) => b.status == 'accepted').toList();
  List<BookingModel> get completedBookings =>
      _bookings.where((b) => b.status == 'completed').toList();

  void listenToStudentBookings(String studentId) {
    _service.getStudentBookings(studentId).listen(
      (list) {
        _bookings = list;
        notifyListeners();
      },
      onError: (_) {
        _error = 'Брондауларды жүктеу мүмкін болмады';
        notifyListeners();
      },
    );
  }

  Future<String?> createBooking(BookingModel booking) async {
    _isLoading = true;
    _error = null;
    notifyListeners();
    try {
      final id = await _service.createBooking(booking);
      _isLoading = false;
      notifyListeners();
      return id;
    } catch (e) {
      _isLoading = false;
      _error = 'Брондау жіберілмеді. Қайталап көріңіз';
      notifyListeners();
      return null;
    }
  }

  Future<bool> deleteBooking(String bookingId) async {
    try {
      await _service.deleteBooking(bookingId);
      _bookings.removeWhere((b) => b.id == bookingId);
      notifyListeners();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> updateStatus(String bookingId, String status) async {
    try {
      await _service.updateBookingStatus(bookingId, status);
      final idx = _bookings.indexWhere((b) => b.id == bookingId);
      if (idx != -1) {
        _bookings[idx] = _bookings[idx].copyWith(status: status);
        notifyListeners();
      }
      return true;
    } catch (_) {
      return false;
    }
  }
}
